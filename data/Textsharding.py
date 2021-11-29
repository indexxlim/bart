from collections import defaultdict
from itertools import islice

import multiprocessing
import statistics


class Sharding:
    def __init__(self, input_files, output_name_prefix, n_training_shards):
        assert len(input_files) > 0, 'The input file list must contain at least one file.'
        assert n_training_shards > 0, 'There must be at least one output shard.'
        assert n_test_shards > 0, 'There must be at least one output shard.'

        self.n_training_shards = n_training_shards
        self.n_test_shards = n_test_shards
        self.fraction_test_set = fraction_test_set

        self.input_files = input_files

        self.output_name_prefix = output_name_prefix
        self.output_training_identifier = '_training'
        self.output_file_extension = '.txt'

        self.articles = {}    # key: integer identifier, value: list of articles
        self.sentences = {}    # key: integer identifier, value: list of sentences
        self.output_training_files = {}    # key: filename, value: list of articles to go into file
        self.output_test_files = {}  # key: filename, value: list of articles to go into file

        self.init_output_files()


    # Remember, the input files contain one article per line (the whitespace check is to skip extraneous blank lines)
    def load_articles(self):
        print('Start: Loading Articles')

        global_article_count = 0
        for input_file in self.input_files:
            print('input file:', input_file)
            with open(input_file, mode='r', newline='\n') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        self.articles[global_article_count] = line.rstrip()
                        global_article_count += 1

        print('End: Loading Articles: There are', len(self.articles), 'articles.')
