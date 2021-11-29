import numpy as np
import torch
import math

class Bart_noising():
    '''
        Make noise from input ids - from bert official and fairseq noising
        https://github.com/google-research/bert
        https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/fairseq/data/noising.py#L39
        https://github.com/pytorch/fairseq/blob/6563407fcb84be52a5cf0e2e64f9230b97271e59/fairseq/data/denoising_dataset.py
    '''
    def __init__(self, tokenizer, mlm_probability=0.15, default_dropout_prob=0.05, default_max_shuffle_distance=4,
                       poisson_lambda = 3):
        self.tokenizer = tokenizer
        self.all_special_ids = self.tokenizer.all_special_ids
        self.vocab_len = len(self.tokenizer.vocab)
        self.mask_token_id = self.tokenizer.mask_token_id


        self.mlm_probability=mlm_probability
        self.default_dropout_prob = default_dropout_prob
        self.default_max_shuffle_distance=default_max_shuffle_distance

        # Set piece_end mapping for non-subword 
        voacb_keys = list(tokenizer.vocab.keys())
        suffixes = ("##", "~~", "~")
        self.piece_end = np.array(
            [
                not (voacb_keys[i].startswith(suffixes) or i in (self.all_special_ids))
                for i in range(len(voacb_keys))
            ]
        )
        self.get_word_idx = (
            self._get_piece_word_idx if self.piece_end is not None else self._get_token_idx
        )

        # Set poisson distribution for infilling
        _lambda = poisson_lambda
        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-_lambda)
        k_factorial = 1
        ps = []
        for k in range(0, 128):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= _lambda
            k_factorial *= (k + 1)
            if ps[-1] < 0.0000001:
                break
        ps = torch.FloatTensor(ps)
        self.mask_span_distribution = torch.distributions.Categorical(ps)


    def noising(self, input_ids):
        if input_ids.size(0) == 1 and input_ids.size(1) == 1:
            return input_ids
            
        lengths = torch.tensor(np.sum(np.logical_not(np.isclose(input_ids, 0)), axis=1))
        
        input_ids,lengths = self.word_dropout(input_ids, lengths) #deletion
        #input_ids,lengths = self.word_shuffle(input_ids, lengths)
        #input_ids = self.masking(input_ids) #bert official
        whole_word_mask_p = self.mlm_probability*0.5
        input_ids,lengths = self.word_dropout(input_ids, lengths, dropout_prob=whole_word_mask_p, blank_idx=self.mask_token_id) #whole_word_masking
        input_ids = self.infilling(input_ids, lengths)
        return input_ids

    def masking(self, input_ids):
        labels = input_ids.clone()
        special_tokens_mask = [[1 if token in self.all_special_ids else 0 for token in input_id] for input_id in input_ids]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability * 0.5)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        #labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)


        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]


        return input_ids

    def _get_piece_word_idx(self, x):
            """
            Given a list of wordpiece tokens, for every index in the tokens list,
            return the index of the word grouping that it belongs to.
            For example, for input x corresponding to ["안녕", "~하", "~세", "~요", "반갑", "~습니다"],
            return [0, 0, 0, 0, 1, 1].
            
            """
            # x: (T x B)
            
            piece_end = self.piece_end[x]

            # do a reduce front sum to generate word ids
            word_idx = piece_end.cumsum(1)
            return word_idx
    
    def _get_token_idx(self, x):
        """
        This is to extend noising functions to be able to apply to non-bpe
        tokens, e.g. word or characters.
        """
        x = torch.t(x)
        word_idx = np.array([range(len(x_i)) for x_i in x])
        return np.transpose(word_idx)

    def word_dropout(self, x, lengths, dropout_prob=None, blank_idx=None):
        if dropout_prob is None:
            dropout_prob = self.default_dropout_prob
        # x: (T x B), lengths: B
        if dropout_prob == 0:
            return x, lengths

        assert 0 < dropout_prob < 1

        # be sure to drop entire words
        word_idx = self.get_word_idx(x)
        sentences = []
        modified_lengths = []
        for i in range(lengths.size(0)):
            num_words = max(word_idx[i,:]) + 1

            has_eos = x[i, lengths[i] - 1] == self.tokenizer.eos_token_id
            if has_eos:  # has eos?
                keep = np.random.rand(num_words - 1) >= dropout_prob
                keep = np.append(keep, [True])  # keep EOS symbol
            else:
                keep = np.random.rand(num_words) >= dropout_prob
            keep[0]=True  # has bos!

            words = x[i,: lengths[i]].tolist()

            # TODO: speed up the following loop
            # drop words from the input according to keep
            new_s = [
                w if keep[word_idx[i,j]] else blank_idx for j, w in enumerate(words)
            ]
            new_s = [w for w in new_s if w is not None]
            # we need to have at least one word in the sentence (more than the
            # start / end sentence symbols)
            if len(new_s) <= 1:
                # insert at beginning in case the only token left is EOS
                # EOS should be at end of list.
                new_s.insert(0, words[np.random.randint(0, len(words))])
            assert len(new_s) >= 1 and (
                not has_eos  # Either don't have EOS at end or last token is EOS
                or (len(new_s) >= 2 and new_s[-1] == self.tokenizer.eos_token_id)
            ), "New sentence is invalid."
            sentences.append(new_s)
            modified_lengths.append(len(new_s))
        # re-construct input
        modified_lengths = torch.LongTensor(modified_lengths)
        modified_x = torch.LongTensor(
            modified_lengths.size(0),modified_lengths.max()
        ).fill_(self.tokenizer.pad_token_id)
        for i in range(modified_lengths.size(0)):
            modified_x[i,: modified_lengths[i]].copy_(torch.LongTensor(sentences[i]))

        return modified_x, modified_lengths
    
    def word_shuffle(self, x, lengths, max_shuffle_distance=None):
        if max_shuffle_distance is None:
            max_shuffle_distance = self.default_max_shuffle_distance
        # x: (T x B), lengths: B
        if max_shuffle_distance == 0:
            return x, lengths

        # max_shuffle_distance < 1 will return the same sequence
        assert max_shuffle_distance > 1

        # define noise word scores
        noise = np.random.uniform(
            0,
            max_shuffle_distance,
            size=(x.size(0), x.size(1)),
        )
        noise[:,0] = -1  # do not move start sentence symbol
        # be sure to shuffle entire words
        word_idx = self.get_word_idx(x)
        x2 = x.clone()
        for i in range(lengths.size(0)):
            length_no_eos = lengths[i]
            if x[i, lengths[i] - 1] == self.tokenizer.eos_token_id:
                length_no_eos = lengths[i] - 1
            # generate a random permutation
            scores = word_idx[i, :length_no_eos] + noise[i, word_idx[i, :length_no_eos]]
            # ensure no reordering inside a word
            scores += 1e-6 * np.arange(length_no_eos.item())
            permutation = scores.argsort()
            # shuffle words
            x2[i,:length_no_eos].copy_(
                x2[i, :length_no_eos][torch.from_numpy(permutation)]
            )
        return x2, lengths
    
    def infilling(self, x, input_lengths, random_ratio=0.1):
        '''
            Data infilling
        '''
        x2 = []
        p = self.mlm_probability * 0.5
    
        #The length before mask and eos
        input_lengths = input_lengths-1
            
        for x_index, source in enumerate(x):
            is_word_start = torch.ByteTensor([0 if token in self.all_special_ids else 1 for token in source])
            num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
            if num_to_mask < 1:
                x2.append(source)
                continue
            num_inserts = 0

            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))
            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask and num_to_mask > 0:
                lengths = torch.cat([lengths, self.mask_span_distribution.sample(sample_shape=(num_to_mask,))], dim=0)
                cum_length = torch.cumsum(lengths, 0)
            # cum_length  == 총 infilling 할 갯수

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]


            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            word_starts = torch.nonzero(is_word_start, as_tuple=False)

            indices = word_starts[torch.randperm(word_starts.size(0))[:num_to_mask]].squeeze(1)
            mask_random = torch.FloatTensor(num_to_mask).uniform_() < random_ratio

            source_length = source.size(0)
            assert source_length - 1 not in indices #eos_token must not be in indices
            to_keep = torch.ones(source_length, dtype=torch.bool)
            is_word_start[input_lengths[x_index]:]= 255 # acts as a long length, so spans don't go over the end of doc

            # keep index, but replace it with [MASK]
            source[indices] = self.tokenizer.mask_token_id
            source[indices[mask_random]] = torch.randint(1, self.vocab_len, size=(mask_random.sum(),))

            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                
                # keep index, but replace it with [MASK]
                source[indices] = self.tokenizer.mask_token_id
                source[indices[mask_random]] = torch.randint(1, self.vocab_len, size=(mask_random.sum(),))
            x2.append(source)
        x2 = torch.stack(x2)

        return x2

