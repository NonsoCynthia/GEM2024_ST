import os
import torch
import argparse
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from datasets import load_dataset, DatasetDict, Dataset
from load_dataset import preprocess_data_
from data.load_dataset import preprocess_data, read_file
from T5 import T5_Model
from training import Trainer
#import pytorch_lightning as pl
from transformers import set_seed
set_seed(42)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_input):
        self.data_input = data_input

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, idx):
        return self.data_input[idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", help="path to the tokenizer")
    parser.add_argument("--model_path", help="path to the model")
    parser.add_argument("--task", help="Traing task")
    parser.add_argument("--data_path", help="path to the data")
    parser.add_argument("--epochs", help="number of epochs", type=int)
    parser.add_argument("--learning_rate", help="learning rate", type=float)
    parser.add_argument("--train_batch_size", help="batch size of training", type=int)
    parser.add_argument("--early_stop", help="earling stop", type=int)
    parser.add_argument("--max_length", help="maximum length to be processed by the network", type=int)
    parser.add_argument("--write_path", help="path to write best model")
    parser.add_argument("--verbose", help="should display the loss?", action="store_true")
    parser.add_argument("--batch_status", help="display of loss", type=int)
    # parser.add_argument("--cuda", help="use CUDA", action="store_true")

    args = parser.parse_args()    

    # Model settings, Settings and configurations
    tokenizer_path = args.tokenizer
    model_path = args.model_path
    task = args.task
    data = args.data_path
    epochs = args.epochs
    learning_rate = args.learning_rate
    train_batch_size = args.train_batch_size
    early_stop = args.early_stop
    max_length = args.max_length
    write_path = args.write_path
    verbose = args.verbose if 'verbose' in args else False
    batch_status = args.batch_status
    # device = torch.device('cuda' if args.cuda else 'cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # input_path = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/Data/deepnlg/"
    # data_input_path = "~/spinning-storage/cosuji/NLG_Exp/gem/gem_data/"

    # Create model
    if "t5" in tokenizer_path.lower():
        mod = 't5'
    elif "ul2" in tokenizer_path.lower():
        mod = 'ul2'
    else:
        raise Exception("Invalid model")

    #if "gem_data" in data:
        #dataset_dict = preprocess_data_(data, task)
        #train_dataset = CustomDataset(dataset_dict["train"])
        #validation_dataset = dataset_dict["validation"]
        #inference_test = dataset_dict["inference_test"]
    #else:

    dataset_dict = preprocess_data(data, task, mod)
    train_data = dataset_dict["train"]
    validation_dataset = dataset_dict["validation"]
    test_dataset = dataset_dict["test"]

    # Concatenate datasets
    concatenated_dataset = ConcatDataset([train_data, validation_dataset, test_dataset])

    # Shuffle the concatenated dataset
    random.seed(42)
    shuffled_indices = list(range(len(concatenated_dataset)))
    random.shuffle(shuffled_indices)

    # Split the shuffled dataset into train and validation datasets
    tr_size = int(len(concatenated_dataset) * 0.8)
    train_indices = shuffled_indices[:tr_size]
    validation_indices = shuffled_indices[tr_size:]

    train_dataset = Subset(concatenated_dataset, train_indices)
    validation_dataset = Subset(concatenated_dataset, validation_indices)

    # Wrap the train dataset with CustomDataset
    train_dataset = CustomDataset(train_dataset)
    
    
    ##Initialize the models
    write_path = os.path.join(write_path, f"{task}/{mod}")
    generator = T5_Model(tokenizer_path, model_path, max_length, sep_token=task+":")

    # Create data loader
    trainloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)#, collate_fn=lambda x:x) #num_workers=10

    # Create optimizer
    optimizer = torch.optim.AdamW(generator.model.parameters(), lr=learning_rate)
    
    # Trainer settings
    trainer = Trainer(generator, trainloader, validation_dataset, optimizer, epochs, batch_status, write_path, early_stop, verbose)

    # Train the model
    trainer.train()
