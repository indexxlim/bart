import os
import argparse
import numpy as np
import collections
import h5py
import tqdm

from bart.tokenization_hanbert import HanBertTokenizer
from google.cloud import storage

class sharding_h5py():
    def __init__(self, tokenizer, num_divided):
        self.tokenizer = tokenizer
        self.num_divided=num_divided
        
        client = storage.Client()
        bucket = storage.Bucket(client, "hanbart_bucket")
        all_blobs = list(client.list_blobs(bucket, prefix='sharding_corpus'))
    
    def make_file(self, input_file, output_file):
        features = collections.OrderedDict()
        features["input_ids"] = np.zeros([self.num_divided, 512], dtype="int32")
        with open(input_file, 'r') as f:
            text = f.readlines()
        for line_index, line in enumerate(tqdm.tqdm(text)):
            a = self.tokenizer.encode_plus(line,
                padding='max_length', #'max_length'
                max_length=512,
                truncation=True,
                #add_special_tokens=False,
                return_tensors='np')['input_ids'][0]
            features["input_ids"][line_index] =a
        
        h5py_f= h5py.File(output_file, 'w')
        h5py_f.create_dataset("input_ids", data=features["input_ids"], dtype='i4', compression='gzip')
        h5py_f.flush()
        h5py_f.close()    


    
def write_single_shard(shard_name, file):
    with open(shard_name, mode='w') as f:
        f.writelines(file)  # Line break between articles    
    
def sharding_text(path, output_path):
    path = 'bart_corpus/'
    output_path ='sharding_corpus/'
    files = os.listdir(path)
    files = sorted(files)
    
    lines = []
    shard_file_index=0
    for file in files:
        print(file)
        f = open(path+file, 'r')
        lines.extend(f.readlines())
        f.close()
        while(len(lines) > num_divided):
            print(len(lines))
            write_single_shard(output_path+f'bartcorpus_{str(shard_file_index).zfill(3)}.txt', lines[:num_divided])
            del lines[:num_divided]
            shard_file_index+=1
        
def sharding_frombucket(bucket_name, folder, output_path):
    from google.cloud import storage
    client = storage.Client()
    bucket = storage.Bucket(client, bucket_name)
    all_blobs = list(client.list_blobs(bucket, prefix=folder))
    
    lines = []
    for blob in all_blobs:
        prinbt(blob.name)
        with all_blobs[0].open("rt") as f:
            lines.extend(f.readlines())

        while(len(lines) > num_divided):
            write_single_shard(output_path+f'bartcorpus_{str(shard_file_index).zfill(3)}.txt', lines[:num_divided])
            del lines[:num_divided]
            shard_file_index+=1

    
def make_all_file():
    '''
        legacy
    '''
    long_number = 0

    num_divided=374439
    file_index = 0
    divided_index=0
    features = collections.OrderedDict()
    features["input_ids"] = np.zeros([num_divided, 512], dtype="int32")



    for file in files:
        print(file)
        f = open(path+file, 'r')
        for line_index, line in enumerate(tqdm(f)):
            if divided_index<num_divided:
                a = tokenizer.encode_plus(line,
                    padding='max_length', #'max_length'
                    max_length=512,
                    truncation=True,
                    #add_special_tokens=False,
                    return_tensors='np')['input_ids'][0]
                features["input_ids"][divided_index] =a
                divided_index+=1
                if a[-1]:
                    long_number+=1


            else:
                print(f"saving data {file_index}th file be filled")

                output_file = f"h5py_corpus/h5py_data_{file_index}"
                h5py_f= h5py.File(output_file, 'w')
                h5py_f.create_dataset("input_ids", data=features["input_ids"], dtype='i4', compression='gzip')
                h5py_f.flush()
                h5py_f.close()

                file_index+=1
                divided_index=0
                features = collections.OrderedDict()
                features["input_ids"] = np.zeros([num_divided, 512], dtype="int32")

        f.close()

    print(f"saving data {file_index}th file be filled")

    output_file = f"h5py_corpus/h5py_data_{file_index}"
    h5py_f= h5py.File(output_file, 'w')
    h5py_f.create_dataset("input_ids", data=features["input_ids"], dtype='i4', compression='gzip')
    h5py_f.flush()
    h5py_f.close()    



    
    

def main():

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus. can be directory with .txt files or a path to a single file")
    parser.add_argument("--output_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The output file where the model checkpoints will be written.")
    
    parser.add_argument("--num_divided",
                        default=None,
                        type=int,
                        required=True,
                        help="How many to divided")

    args = parser.parse_args()
    
    tokenizer = HanBertTokenizer.from_pretrained("bart/HanBart-54kN")
    
    shard_class = sharding_h5py(tokenizer, args.num_divided)
    
    print(f'wrtie in {args.output_file}')
    
    shard_class.make_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()    