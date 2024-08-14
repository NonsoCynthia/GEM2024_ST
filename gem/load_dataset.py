import os
import re
import torch
import numpy as np
import json
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import DatasetDict, Dataset


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Example usage:
# SEED = 42
# set_seed(SEED)

# Read data from files
def read_file(path):
    if path.endswith(".json"):
        with open(path, "r", encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    else:
        with open(path, 'r', encoding='utf-8') as file:
            contents = file.read()
            contents = contents.replace('<', '[').replace('>', ']')
            lines = [line.strip() for line in contents.split('\n')]
            if lines and lines[-1] == '':
                return lines[:-1]
            return lines

def preprocess_data_(path, task):
    data = pd.read_json(os.path.join(path, "webnlg_17_data.json"), orient="records")
    data = data.sample(frac=1).reset_index(drop=True)

    inference_data = pd.read_json(os.path.join(path, "inference.json"), encoding='utf-8-sig', lines=True, orient="records")

    train_size = int(len(data) * 0.9)
    train = data[:train_size]
    dev = data[train_size:]

    if task == "ordering":
        train_df = pd.DataFrame({"Source": train["order_in"], "Target": train["order_out"]})
        dev_df = pd.DataFrame({"Source": dev["order_in"], "Target": dev["order_out"]})
        inference_test = pd.DataFrame({"Source": inference_data["triples"]})

    else:
        train_df = pd.DataFrame({"struct_in": train["struct_in"], "struct_out": train["struct_out"]})
        dev_df = pd.DataFrame({"struct_in": dev["struct_in"], "struct_out": dev["struct_out"]})
        inference_test = pd.DataFrame({"Source": inference_data["triples"]})

    dataset = {
        "train": pd.DataFrame(train_df),
        "validation": pd.DataFrame(dev_df),
        "inference_test": pd.DataFrame(inference_test)
    }

    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(dataset["train"]),
        "validation": Dataset.from_pandas(dataset["validation"]),
        "inference_test": Dataset.from_pandas(dataset["inference_test"])
    })

    return dataset_dict



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_input):
        self.data_input = data_input

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, idx):
        return self.data_input[idx]

# # Usage
#path = "~/spinning-storage/cosuji/NLG_Exp/gem/gem_data/"
#path =  # "/content/output.json"
#task = "ordering"
#dataset_dict = preprocess_data_(path, task)
# # Example usage
#train_dataset = CustomDataset(dataset_dict["train"])
#print(len(train_dataset['Source']), len(train_dataset['Target']))
#val_dataset = CustomDataset(dataset_dict["validation"])
#print(len(val_dataset['Source']), len(val_dataset['Target']))
#inference_dataset = CustomDataset(dataset_dict["inference_test"])
#print(len(inference_dataset['Source']))
# print(train_dataset['Source'][:5])
