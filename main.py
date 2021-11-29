'''
    HanBart Training
'''

import datetime
import os
import re
import logging
import argparse
import torch

from configloader import train_config
import transformers
from dataloader import KoreanDataset, KoreanBatchGenerator, get_dataloader
from tokenization_hanbart import HanBartTokenizer
from train import train

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

    checkpoints = map(lambda x: int(re.search(r"[0-9]{1,}", x).group()[0]),files)

    try:
        last_checkpoint = sorted(checkpoints)[-1]
    except:
        last_checkpoint = 0
    return last_checkpoint

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
        "--device",
        type=str,
        default=train_config.setup.device
    )
    parser.add_argument(
        "--tpu",
        type=str,
        default=train_config.setup.device
    )
    parser.add_argument(
        "--xla_parallel",
        type=str,
        default=train_config.setup.device
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
    args.device = xm.xla_device() if args.tpu else args.device if args.device else 'cpu'

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
    
    # Define Model
    model_class = getattr(transformers, args.model_class)

    if os.path.isdir(args.checkpoint):
        args.checkpoint_count = checkpoint_count(args.checkpoint)
        logger = get_logger(args)
        logger.info(f"Checkpoint path directory exists")
        logger.info(f"Loading model from saved_checkpoint_{args.checkpoint_count}")
        model = torch.load(f"{args.checkpoint}/saved_checkpoint_{args.checkpoint_count}")

        args.checkpoint_count +=1

    else:
        try:
            os.makedirs(args.checkpoint)
        except:
            print("Ignoring Existing File Path ...")

        model = model_class.from_pretrained(args.config_class)
        args.checkpoint_count = 0
        logger = get_logger(args)

        logger.info(f"Model Creation")
        logger.info(f"Checkpoint creation {args.checkpoint}")

    args.logger = logger

    model.to(args.device)

    tokenizer = HanBertTokenizer.from_pretrained(args.tokenizer)

    
    optimizer_class = getattr(transformers, args.optimizer_class)
    optimizer = optimizer_class(model.parameters(), lr=args.learning_rate)

    logger.info(f"Loading data from {args.data_dir} ...")

    dataset = KoreanDataset(data_path=os.path.join(args.data_dir, "bart_corpus_some.txt"))

    batchgenerator = KoreanBatchGenerator(tokenizer, make_noise=True)
    train_loader = get_dataloader(dataset, batchgenerator, shuffle=False)

    train(model, optimizer, tokenizer, train_loader, None, None, args)

if __name__ == "__main__":
    main()
