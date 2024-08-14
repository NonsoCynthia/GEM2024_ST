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
        try:
            with open(path, "r", encoding='utf-8') as json_file:
                data = json.load(json_file)
        except:
            json_string = open(path).read()
            data = json.loads(json_string)
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
    ftask = ["factual", "fictional", "counterfactual", "simon"]
    factual_data = pd.read_json(os.path.join(path, f"{ftask[0]}.json"), encoding='utf-8-sig', lines=True, orient="records")
    fictional_data = pd.read_json(os.path.join(path, f"{ftask[1]}.json"), encoding='utf-8-sig', lines=True, orient="records")
    counterfactual_data = pd.read_json(os.path.join(path, f"{ftask[2]}.json"), encoding='utf-8-sig', lines=True, orient="records")
    simon_data = pd.read_json(os.path.join(path, f"{ftask[3]}.json"), encoding='utf-8-sig', lines=True, orient="records")

    
    #THIS WILL THROW UP AN ERROR BECAUSE THE FILE HAS NOT BEEN CREATED YET
    #factual_struct = read_file(f"/home/cosuji/spinning-storage/cosuji/NLG_Exp/gem/results/ordering/{ftask[0]}_ordering_test.mapped")
    #fictional_struct = read_file(f"/home/cosuji/spinning-storage/cosuji/NLG_Exp/gem/results/ordering/{ftask[1]}_ordering_test.mapped")
    #counterfactual_struct = read_file(f"/home/cosuji/spinning-storage/cosuji/NLG_Exp/gem/results/ordering/{ftask[2]}_ordering_test.mapped")

    if task == "ordering":
        factual_df = pd.DataFrame({"Source": factual_data['triples']})
        fictional_df = pd.DataFrame({"Source": fictional_data["triples"]})
        counterfactual_df = pd.DataFrame({"Source": counterfactual_data['triples']})
        simon_df = pd.DataFrame({"Source": simon_data['triples']})

    else:
        factual_df = pd.DataFrame({"Source": factual_struct})
        fictional_df = pd.DataFrame({"Source": fictional_struct})
        counterfactual_df = pd.DataFrame({"Source": counterfactual_struct})
        simon_df = pd.DataFrame({"Source": simon_struct})

    dataset = {
        "factual": pd.DataFrame(factual_df),
        "fictional": pd.DataFrame(fictional_df),
        "counterfactual": pd.DataFrame(counterfactual_test),
        "simon": pd.DataFrame(simon_test)
    }

    dataset_dict = DatasetDict({
        "factual": Dataset.from_pandas(dataset["factual"]),
        "fictional": Dataset.from_pandas(dataset["fictional"]),
        "counterfactual": Dataset.from_pandas(dataset["counterfactual"]),
        "simon": Dataset.from_pandas(dataset["simon"])
    })

    return dataset_dict



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_input):
        self.data_input = data_input

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, idx):
        return self.data_input[idx]



def preprocess_data(path, task=None):
    ftask = ["factual", "fictional", "counterfactual", "simon"]
    data = {}
    
    for t in ftask:
        if task == "ordering":
            data[t] = pd.read_json(os.path.join(path, f"{t}.json"), encoding='utf-8-sig', lines=True, orient="records")
        elif task == 'structuring':
            struct_path = '/home/cosuji/spinning-storage/cosuji/NLG_Exp/gem/results/ordering'
            struct_file = read_file(os.path.join(struct_path, f"{t}_ordering.mapped"))
            data[t] = pd.DataFrame({"triples": struct_file})
        else:
            data[t] = pd.read_json(os.path.join(path, f"{t}.json"), encoding='utf-8-sig', lines=True, orient="records")
            #struct_path = '/home/cosuji/spinning-storage/cosuji/NLG_Exp/gem/results/structuring'
            struct_file = read_file(os.path.join(path, f"{t}_struct.txt"))
            data[f"{t}_struct"] = pd.DataFrame({"triples": struct_file})

    dataset = {}
    
    for t, df in data.items():
        if task == 'ordering':
            dataset[t] = Dataset.from_pandas(pd.DataFrame({"Source": df['triples']}))
        elif task == 'structuring':
            dataset[t] = Dataset.from_pandas(pd.DataFrame({'Source': df['triples']}))
        else:
            dataset[t] = Dataset.from_pandas(pd.DataFrame({'Source': df['triples']}))

    return DatasetDict(dataset)



#DatasetDict({
    #factual: Dataset({
        #features: ['Source'],
        #num_rows: 1779
    #})
    #factual_struct: Dataset({
        #features: ['Source'],
        #num_rows: 1779
    #})
    #fictional: Dataset({
        #features: ['Source'],
        #num_rows: 1779
    #})
    #fictional_struct: Dataset({
        #features: ['Source'],
        #num_rows: 1779
    #})
    #counterfactual: Dataset({
        #features: ['Source'],
        #num_rows: 1779
    #})
    #counterfactual_struct: Dataset({
        #features: ['Source'],
        #num_rows: 1779
    #})
#})




# # Usage
#path = "~/spinning-storage/cosuji/NLG_Exp/gem/gem_data/"
#task = None
#dataset_dict = preprocess_data(path, task)
#print(dataset_dict)
# # Example usage
#train_dataset = CustomDataset(dataset_dict["train"])
#print(len(train_dataset['Source']), len(train_dataset['Target']))
#val_dataset = CustomDataset(dataset_dict["validation"])
#print(len(val_dataset['Source']), len(val_dataset['Target']))
#inference_dataset = dataset_dict["factual_struct"]
#print(len(inference_dataset['Source']))
#item = [287, 453, 1161, 1749]
#for i in item:
    #print(i-1, '==', inference_dataset['Source'][i-1])
