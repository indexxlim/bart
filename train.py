'''
    REFERENCE: BART Trainer
'''
import json
import time
import os
import random
import tqdm
from tqdm.utils import disp_len

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.debug.metrics as met
from transformers import set_seed
from dataloader import h5pyDataset, h5pyBatchGenerator, get_dataloader


SEED = 42
set_seed(SEED)

def single_epoch_train(model, optimizer, train_loader, pad_token_id, args, scheduler):
    #model.train()
    logger = args.logger
    loader = tqdm.tqdm(train_loader, total=int(len(train_loader)))
    device = args.device

    if args.xla_parallel:
        loader = pl.MpDeviceLoader(train_loader, device)

    for idx, batch in enumerate(loader):

        input_ids, attention_mask, decoder_input_ids, labels = (
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['decoder_input_ids'].to(device),
            batch['labels'].to(device)
        )

        #labels[labels == pad_token_id] = -100 # in dataloader


        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_input_ids = decoder_input_ids,
            labels = labels
        )
        loss = outputs[0]


        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        if idx % args.gradient_accumulation_steps:
            if args.xla_parallel:
                xm.optimizer_step(optimizer)
                if scheduler:
                    scheduler.step()
            
            elif args.tpu:
                optimizer.step()
                xm.mark_step()

            else:
                optimizer.step()
            model.zero_grad()


        if not idx % args.log_every:
            #xm.master_print(f"Batch Loss: {loss.item():.3f} - {idx}/{len(loader)}")
            logger.info(f"Loss: {loss.item()*args.gradient_accumulation_steps} - {idx}/{len(loader)}")
        #xm.master_print(met.metrics_report())



def single_epoch_validate(model, tokenizer, valid_loader, pad_id, args):

    model.eval()
    loader = tqdm.tqdm(valid_loader)
    device = args.device

    oupput = []
    losses = []
    for idx, batch in enumerate(loader):
        input_ids, decoder_input_ids, labels = (
            batch['input_ids'].to(device),
            batch['decoder_input_ids'].to(device),
            batch['labels'].to(device)
        )

def train(model, optimizer, tokenizer, train_loader, valid_loader, test_loader, args, data_name):
    logger = args.logger
    

    for epoch in range(args.epochs):

        start_time = time.time()

        logger.info(f"Epoch {epoch + 1} (Globally {args.checkpoint_count})")

        #Training
        logger.info(f"Begin Training ...")
        single_epoch_train(model , optimizer, train_loader, tokenizer.pad_token_id, args)
        mins = round((time.time() - start_time) / 60, 2)
        logger.info(f"Training Finished!")
        logger.info(f"Time taken for training epoch {epoch+1} (globally {args.checkpoint_count}):{mins} mins(s)")


        
        # Saving model
        #if args.checkpoint_count % 2 == 0:
        #model.save_pretrained(f"{args.checkpoint}/saved_checkpoint_{args.checkpoint_count}_{data_name}")
        xm.save(model.state_dict(), f"{args.checkpoint}/model_{args.checkpoint_count}_{data_name}.bin")
        logger.info(f"Checkpoint saved at {args.checkpoint}/saved_checkpoint_{args.checkpoint_count}_{data_name}")


        args.checkpoint_count += 1



def train_multiple(model, optimizer, tokenizer, args, scheduler=None):
    logger = args.logger

    file_list = os.listdir(args.data_dir)
    sorted(file_list)

    
    for epoch in range(args.epoch, args.epochs):

        logger.info(f"Epoch {epoch + 1} (Globally {args.checkpoint_count})")
        logger.info(f"Begin Training ...")



        for data_name in file_list:
            if data_name in args.trained_data:
                continue

            logger.info(f"Loading data {data_name} ...")
            
            # if not xm.is_master_ordinal():
            #     xm.rendezvous('read_only_once')

            dataset = h5pyDataset(data_path=os.path.join(args.data_dir, data_name))

            # if xm.is_master_ordinal():
            #     xm.rendezvous('read_only_once')

            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True
            )

            batchgenerator = h5pyBatchGenerator(tokenizer, make_noise=True)
            train_loader = get_dataloader(dataset, batchgenerator, shuffle=False, batch_size=args.train_batch_size, sampler=train_sampler)


            start_time = time.time()


            #Training
            single_epoch_train(model , optimizer, train_loader, tokenizer.pad_token_id, args, scheduler)
            mins = round((time.time() - start_time) / 60, 2)
            logger.info(f"{data_name} Time taken for training epoch {epoch+1} (globally {args.checkpoint_count}):{mins} mins(s)")

            args.trained_data.append(data_name)
            
            # Saving model
            #model.save_pretrained(f"{args.checkpoint}/saved_checkpoint_{args.checkpoint_count}_{data_name}")
            xm.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'trained_data_list': args.trained_data,
                    'scheduler': scheduler.state_dict()
                    }, f"{args.checkpoint}/saved_checkpoint_{args.checkpoint_count}.ckpt")
            logger.info(f"Checkpoint saved at {args.checkpoint}/saved_checkpoint_{args.checkpoint_count}")


            args.checkpoint_count += 1

        logger.info(f"{epoch}th epoch Training Finished!")
        args.trained_data=[]
            

