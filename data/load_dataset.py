import os
import re
import torch
import numpy as np
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict, Dataset

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Read data from files
def read_file(path):
    if path.endswith(".json"):
        with open(path, "r", encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    else:
        with open(path, 'r', encoding='utf-8') as file:
            contents = file.read()
            #contents = contents.replace('</TRIPLE> <TRIPLE>',',').replace('<TRIPLE>', '').replace('</TRIPLE>', '')
            contents = contents.replace('<', '[').replace('>', ']')
            contents = contents.replace('...','.').replace('..','.')
            lines = [line.strip() for line in contents.split('\n')]
            if lines and lines[-1] == '':
                return lines[:-1]
            return lines
        

def preprocess_data(path, task, model):
    task_suffix = { 
        "ordering": "eval",
        "structuring": "ordering",  # Assuming "structuring" uses "ordering" data
        "sr": "structuring",  # Assuming "REG" uses "lexicalisation" data
    }.get(task, "sr")  # Default to ".reg" if task is not recognized


    if task == "sr":
        #path = '/home/cosuji/spinning-storage/cosuji/NLG_Exp/gem/data/webnlg_17_data.json'
        #train_df = process_data(os.path.join(path, f"input/{task}/train.json"))
        #dev_df = process_data(os.path.join(path, f"input/{task}/dev.json"))
        #test_df = process_data(os.path.join(path, f'input/{task}/test.json'))
        pass
    else:
        train_df = pd.DataFrame({
            "Source": read_file(os.path.join(path, f"input/{task}/train.src")),
            "Target": read_file(os.path.join(path, f"input/{task}/train.trg"))
        })
        train_df = train_df.sample(frac=1).reset_index(drop=True)

        dev_df = pd.DataFrame({
            "Source": read_file(os.path.join(path, f"input/{task}/dev.eval")),
            "Target": read_file(os.path.join(path, f"input/{task}/references/dev.trg1"))
        })

        test_df = pd.DataFrame({
            "Source": read_file(os.path.join(path, f"input/{task}/test.eval")),
            "Target": read_file(os.path.join(path, f"input/{task}/references/test.trg1"))
        })


    dataset = {
        "train": pd.DataFrame(train_df),
        "validation": pd.DataFrame(dev_df),
        "test": pd.DataFrame(test_df)
    }

    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(dataset["train"]),
        "validation": Dataset.from_pandas(dataset["validation"]),
        "test": Dataset.from_pandas(dataset["test"])
    })
        #[]
    #return dataset if model == 'llama2' else dataset_dict
    return dataset_dict 


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_input):
        self.data_input = data_input

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, idx):
        return self.data_input[idx]


# # Usage
# path = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/data/deepnlg/"
# task = "sr"
# model = "t5"
# dataset_dict = preprocess_data(path, task, model)
# # Example usage
# train_dataset = CustomDataset(dataset_dict["pipeline_eval"])
# # print(len(train_dataset['Source']), len(train_dataset['Target'])
# print(len(train_dataset['Source']))
