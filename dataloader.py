'''
    REFERENCE: BART Data loader
'''
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
from noise_function import Bart_noising
import copy
import torch.nn.functional as F
import h5py

def shift_tokens_right(input_ids, pad_token_id, eos_token_id):
    """
        Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
        decoder input: <eos><sos> tok1 tok2 … tokn
        target:<sos> tok1 tok2 … tokn <eos>
        https://github.com/huggingface/transformers/issues/7961
    """
    prev_output_tokens = input_ids.clone()
    #index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = torch.tensor(eos_token_id)
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    
    # last_index = prev_output_tokens.ne(pad_token_id).sum(dim=1)-1
    # for i, last_i in enumerate(last_index):
    #     prev_output_tokens[i][last_i] = pad_token_id
    
    

    assert pad_token_id is not None, "self.tokenizer.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    return prev_output_tokens

class KoreanDataset(Dataset):
    '''
        To read Korean Corpus
    '''
    def __init__(self,data_path):
        with open(data_path, 'r') as f:
            self.data = f.readlines()

    def __getitem__(self, index):       
        return {'context':self.data[index]}
        
    def __len__(self):
        return len(self.data)    

class KoreanDataset4tcs(Dataset):
    '''
        To read Korean Corpus
    '''
    def __init__(self,data_path, tokenizer, bucket):
        blob = bucket.get_blob('bart_corpus/bartip-1.selected.txt')
        with blob.open("rt") as f:
            self.data = f.readlines()

    def __getitem__(self, index):       
        return {'context':self.data[index]}
        
    def __len__(self):
        return len(self.data)  

class KoreanBatchGenerator:
    '''
        collate function
    '''
    def __init__(self, tokenizer, make_noise=True, mlm_probability=0.15, default_dropout_prob=0.1):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.noise = make_noise
        self.noise_func = Bart_noising(tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, batch):

        context = [item['context'] for item in batch]


        source_batch = self.tokenizer.batch_encode_plus(context,
                                                        padding='max_length', #'max_length'
                                                        max_length=512,
                                                        truncation=True,
                                                        #add_special_tokens=False,
                                                        return_tensors='pt')
        input_labels = source_batch.input_ids.clone()
        if self.noise:
            input_ids = self.noise_func.noising(input_labels.clone())

        
        input_labels[input_labels == self.pad_token_id] = -100

        input_ids = F.pad(input=input_ids, pad=(0, 512-input_ids.shape[1]), mode='constant', value=0)

        feature = {'input_ids': input_ids,
                'attention_mask': input_ids.ne(self.pad_token_id)*1,
                'decoder_input_ids': shift_tokens_right(source_batch.input_ids, self.pad_token_id, self.eos_token_id),
                'labels': input_labels}
        #feature['label'] = [item['label'] for item in batch] if 'label' in batch[0] else None
            
        return feature

class h5pyDataset(Dataset):
    '''
        To read Korean Corpus
    '''
    def __init__(self,data_path):
        f = h5py.File(data_path, "r")
        self.data = torch.from_numpy(f['input_ids'][:]).long()
        f.close()

    def __getitem__(self, index):       
        return self.data[index]
        
    def __len__(self):
        return len(self.data)               


class h5pyBatchGenerator:
    '''
        collate function
    '''
    def __init__(self, tokenizer, make_noise=True, mlm_probability=0.15, default_dropout_prob=0.1):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.noise = make_noise
        self.noise_func = Bart_noising(tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, source_batch):
        source_batch = torch.stack(source_batch)

        input_labels = source_batch.clone()
        if self.noise:
            input_ids = self.noise_func.noising(input_labels.clone())

        
        input_labels[input_labels == self.pad_token_id] = -100

        input_ids = F.pad(input=input_ids, pad=(0, 512-input_ids.shape[1]), mode='constant', value=0)

        feature = {'input_ids': input_ids,
                'attention_mask': input_ids.ne(self.pad_token_id)*1,
                'decoder_input_ids': shift_tokens_right(source_batch, self.pad_token_id, self.eos_token_id),
                'labels': input_labels}
        #feature['label'] = [item['label'] for item in batch] if 'label' in batch[0] else None
            
        return feature




def get_dataloader(dataset, batch_generator, batch_size=4, shuffle=True, sampler=None):
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             collate_fn=batch_generator,
                             num_workers=4,
                             sampler=sampler,
                             prefetch_factor=16)
    return data_loader

                        

if __name__ =="__main__":
    print('main')
