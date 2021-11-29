'''
    HanBart Training
'''

import datetime
import os
import re
import logging
import json
import argparse
import torch
from argparse import Namespace


from configloader import train_config
import transformers
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer, AdamW
from dataloader import h5pyDataset, h5pyBatchGenerator, get_dataloader
from tokenization_hanbert import HanBertTokenizer
from transformers.optimization import Adafactor, AdafactorSchedule
from train import train, train_multiple
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.serialization as xser




def gen_checkpoint_id(args):
    model_id = args.MODEL_ID
    timez = datetime.datetime.now().strftime("%Y%m%d%H%M")
    checkpoint_id = "_".join([model_id, timez])
    return checkpoint_id

def get_logger(args):
    log_path = f"{args.checkpoint}/info"

    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    train_instance_log_files = os.listdir(log_path)
    train_instance_count = len(train_instance_log_files)


    # def timetz(*args):
    #     return datetime.now(tz).timetuple()

    # tz = timezone('Asia/Seoul') # UTC, Asia/Shanghai, Europe/Berlin

    # logging.Formatter.converter = timetz


    logging.basicConfig(
        filename=f'{args.checkpoint}/info/train_instance_{train_instance_count}_info.log',
        filemode='w',
        format="%(asctime)s | %(filename)15s | %(levelname)7s | %(funcName)10s | %(message)s",
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logger.info("-"*40)
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info("-"*40)

    return logger

def checkpoint_count(checkpoint):
    _, folders, files = next(iter(os.walk(checkpoint)))
    files = list(filter(lambda x :"saved_checkpoint_" in x, files))

    checkpoints = map(lambda x: int(re.search(r"[0-9]{1,}", x).group()),files)

    try:
        last_checkpoint = sorted(checkpoints)[-1]
    except:
        last_checkpoint = 0
    return last_checkpoint

def serialize_args(args):
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except:
            return False
    
    dct = {k: v for k, v in args.__dict__.items() if is_jsonable(v) }
    return dct    

def model_load(ckpt_file):
    ckpt = torch.load(ckpt_file)
    epoch = ckpt['epoch']
    model_state_dict = ckpt['model_state_dict']
    optimizer_state_dict = ckpt['optimizer_state_dict']
    trained_data = ckpt['trained_data_list']

    return  epoch, model_state_dict, optimizer_state_dict, trained_data    

def get_args():
    global train_config

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--MODEL_ID",
        type=str,
        default=train_config.tbai_required.model_id
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default=train_config.setup.model_class
    )
    parser.add_argument(
        "--tokenizer_class",
        type=str,
        default=train_config.setup.tokenizer_class
    )
    parser.add_argument(
        "--optimizer_class",
        type=str,
        default=train_config.setup.optimizer_class
    )
    parser.add_argument(
        "--config_class",
        type=str,
        default=train_config.setup.config_class
    )    
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=train_config.setup.tokenizer
    )
    parser.add_argument(
        "--model",
        type=str,
        default=train_config.setup.model
    )
    parser.add_argument(
        "--device",
        type=str,
        default=train_config.setup.device
    )
    parser.add_argument(
        "--bucket_name",
        type=str,
        default=train_config.setup.bucket_name
    )
    parser.add_argument(
        "--tpu",
        type=str,
        default=train_config.setup.tpu
    )
    parser.add_argument(
        "--xla_parallel",
        type=str,
        default=train_config.setup.xla_parallel
    )    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=train_config.setup.checkpoint if hasattr(train_config.setup, "checkpoint") else None
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=train_config.hyperparameters.learning_rate
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=train_config.hyperparameters.epochs
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=train_config.hyperparameters.train_batch_size
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=train_config.hyperparameters.eval_batch_size
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=train_config.hyperparameters.gradient_accumulation_steps
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=train_config.hyperparameters.log_every
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=train_config.setup.data_dir
    )
    args=parser.parse_args()
    args.device = args.device if args.device else 'cpu'


    return args
    
def main():
    # Get ArgParse  
    args=get_args()
    if args.checkpoint:
        args.checkpoint = (
            "./model_checkpoint/" + args.checkpoint[-1]
            if args.checkpoint[-1] == "/"
            else "./model_checkpoint/" + args.checkpoint
        )
    else:
        args.checkpoint = "./model_checkpoint/" + gen_checkpoint_id(args)
        


    if os.path.isdir(args.checkpoint):
        args.checkpoint_count = checkpoint_count(args.checkpoint)
        logger = get_logger(args)
    else:
        try:
            os.makedirs(args.checkpoint)
        except:
            print("Ignoring Existing File Path ...")
        args.checkpoint_count = 0
        logger = get_logger(args)
    args.logger = logger

    with open(f"{args.checkpoint}/args.json", "w") as f:
        json.dump(serialize_args(args), f)
    
    
    if args.xla_parallel:
        args = vars(args)
        xmp.spawn(_mp_fn, args=(args,), nprocs=8, start_method='fork')
    else:    
    # Define Model
        model_class = getattr(transformers, args.model_class)            

        config = BartConfig.from_json_file("HanBart-54kN/config.json")

        model = model_class(config)

        args.logger.info(f"Model Creation")
        args.logger.info(f"Checkpoint creation {args.checkpoint}")

        model.to(args.device)

        tokenizer = HanBertTokenizer.from_pretrained(args.tokenizer)

        
        optimizer_class = getattr(transformers, args.optimizer_class)
        #if args.optimizer_class == "Adafactor":
        optimizer = optimizer_class(model.parameters(), warmup_init=True)
#        else:
            #optimizer = optimizer_class(model.parameters(), lr=args.learning_rate * xm.xrt_world_size())
        lr_scheduler = AdafactorSchedule(optimizer)


        logger.info(f"Loading data from {args.data_dir} ...")

        file_list = os.listdir(args.data_dir)
        #for data_name in file_list:
        data_name="h5py_data_0"

        dataset = h5pyDataset(data_path=os.path.join(args.data_dir, data_name))

        batchgenerator = h5pyBatchGenerator(tokenizer, make_noise=True)
        train_loader = get_dataloader(dataset, batchgenerator, batch_size=args.train_batch_size)

        train(model, optimizer, tokenizer, train_loader, None, None, args, data_name, lr_scheduler)


def _mp_fn(index, args):
    args = Namespace(**args)

    args.device = xm.xla_device() if args.tpu else args.device
    logger = args.logger


    model_class = getattr(transformers, args.model_class)

    config = BartConfig.from_json_file("HanBart-54kN/config.json")
    model = model_class(config)


    model.to(args.device)

    tokenizer = HanBertTokenizer.from_pretrained(args.tokenizer)
    optimizer_class = getattr(transformers, args.optimizer_class)
    
    #if args.optimizer_class == "Adafactor":
    optimizer = optimizer_class(model.parameters(), warmup_init=True)
    #else:
        #optimizer = optimizer_class(model.parameters(), lr=args.learning_rate * xm.xrt_world_size())
    lr_scheduler = AdafactorSchedule(optimizer)


#    args.epoch = 0
#    args.trained_data = []
    if args.checkpoint_count>0:
        ckpt_file = f"{args.checkpoint}/saved_checkpoint_{args.checkpoint_count}.ckpt"

        epoch, model_state_dict, optimizer_state_dict, trained_data = model_load(ckpt_file)
        args.epoch = epoch
        args.trained_data = trained_data

        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)


        logger.info(f"{args.checkpoint_count}th Model Loading ")
        args.checkpoint_count +=1
        logger.info(f"Checkpoint creation {args.checkpoint}")


    else:
        args.epoch = 0
        args.trained_data = []

    model.train().to(args.device)

    logger.info(f"Model Creation")
    logger.info(f"Checkpoint creation {args.checkpoint}")



    train_multiple(model, optimizer, tokenizer, args, lr_scheduler)


if __name__ == "__main__":
    main()
