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
            contents = contents.replace('</TRIPLE> <TRIPLE>',',').replace('<TRIPLE>', '').replace('</TRIPLE>', '')
            contents = contents.replace('...','.').replace('..','.')
            lines = [line.strip() for line in contents.split('\n')]
            if lines and lines[-1] == '':
                return lines[:-1]
            return lines
        
def read_file_map(path):
    if path.endswith(".json"):
        with open(path, "r", encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    else:
        with open(path, 'r', encoding='utf-8') as file:
            contents = file.read()
            # Do not replace `<` and `>` if they are not part of your tags
            contents = contents.replace('<', '[').replace('>', ']')
            xcontents = contents.replace('[ /SNT)', '[/SNT]').replace('[/SNT)', '[/SNT]').replace('[/ SNT] ', '[/SNT]')
            xcontents = xcontents.replace('[/ Snt] ', '[/SNT]').replace('[/sNT)', '[/SNT]').replace('[/T] ', '[/SNT]')
            xcontents = xcontents.replace('[SNT?]', '[SNT]').replace('/SOD', '/SNT').replace('[ SNT]', '[SNT]')
            xcontents = xcontents.replace('/SNT]', '[/SNT]')
            xcontents = xcontents.replace('[[[', '[').replace('[[', '[')        #.replace('[[/TRIPLE]', '[/TRIPLE]')
            # xcontents = xcontents.replace('<', '[').replace('>', ']')
            lines = [line.strip() for line in xcontents.split('\n')]
            # print(lines)
            if lines and lines[-1] == '':
                return lines[:-1]
            return lines


def read_dict(file_path):
    with open(file_path, 'r') as file:
        dictionary = json.load(file)
    return dictionary

def realize_date(entity):
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    date_formats = [
        r'(\d{4})-(\d{2})-(\d{2})',
        r'(\d{4})-(\d{1})-(\d{1})',
        r'(\d{2})-(\d{2})-(\d{4})',
        r'(\d{1})-(\d{1})-(\d{4})'
    ]

    for date_format in date_formats:
        match = re.match(date_format, entity)
        if match:
            groups = match.groups()
            year, month, day = map(int, groups[-3:])
            # print(f'day:{day}, month:{month}, year:{year}')
            month_name = month_names.get(month, '')

            if month_name:
                if len(groups[0]) == 4:
                    year, day = day, year
                    formatted_date = f'{month_name} {year}, {day}'
                else:
                    # year, day = day, year
                    formatted_date = f'{month_name} {year}, {day}'

                return True, formatted_date

    return False, ''

def process_data(file_path):
    data = read_file(file_path)

    src = []
    trg = []

    for entry in data:
        # Concatenate fields and convert them to strings if they are lists
        pre_context_str = ' '.join(map(str, entry['pre_context']))
        pos_context_str = ' '.join(map(str, entry['pos_context']))
        entity_str = str(entry['entity'])
        # refex_str = str(entry['refex'])
        refex_str = ' '.join(map(str, entry['refex']))
        src.append(pre_context_str + ' ' +pos_context_str +' ['+ entity_str+']')
        isDate, refex = realize_date(refex_str)
        trg.append(refex_str if not isDate else refex)

    dataframe = pd.DataFrame({"Source": src, "Target": trg})
    return dataframe


def preprocess_data(path, task, model):
    task_suffix = {
        "end2end": "eval", 
        "ordering": "eval",
        "structuring": "ordering",  # Assuming "structuring" uses "ordering" data
        "lexicalization": "structuring",  # Assuming "lexicalisation" uses "structuring" data
        "reg": "lexicalization",  # Assuming "REG" uses "lexicalisation" data
        "sr": "reg"  # Assuming "sr" uses "REG" data
    }.get(task, "reg")  # Default to ".reg" if task is not recognized


    if task == "reg":
        train_df = process_data(os.path.join(path, f"input/{task}/train.json"))
        dev_df = process_data(os.path.join(path, f"input/{task}/dev.json"))
        test_df = process_data(os.path.join(path, f"input/{task}/test.json"))
    elif task == "sr":
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


    # Pipelining Dataset
    if task == "reg":
        path = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results"
        dev_ordering = read_file(os.path.join(path, f"ordering/{model}/dev.ordering.mapped"))
        test_ordering = read_file(os.path.join(path, f"ordering/{model}/test.ordering.mapped"))
        test_lexicalization = read_file_map(os.path.join(path, f"{task_suffix}/{model}/{task_suffix}_pipeline_test.txt"))
        dev_lexicalization = read_file_map(os.path.join(path, f"{task_suffix}/{model}/{task_suffix}_pipeline_eval.txt"))

        pipeline_eval = pd.DataFrame({
            "Source": dev_ordering,
            "Target": dev_lexicalization
        }) 

        pipeline_test = pd.DataFrame({
            "Source": test_ordering,
            "Target": test_lexicalization
        })

    else:
        if task == "ordering" or task == "end2end":
            goto_dev = os.path.join(path, f"results/dev.{task_suffix}")
            goto_test = os.path.join(path, f"results/test.{task_suffix}")

            # pipeline_eval = pd.DataFrame({"Source": read_file(goto_dev)})
            # pipeline_test = pd.DataFrame({"Source": read_file(goto_test)})

        else:
            path = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results"
            if task == 'sr':
                goto_dev = os.path.join(path, f"{task_suffix}/{model}/{task_suffix}_pipeline_eval.txt")
                goto_test = os.path.join(path, f"{task_suffix}/{model}/{task_suffix}_pipeline_test.txt")
            else:
                goto_dev = os.path.join(path, f"{task_suffix}/{model}/dev.{task_suffix}.mapped")
                goto_test = os.path.join(path, f"{task_suffix}/{model}/test.{task_suffix}.mapped")

        pipeline_eval = pd.DataFrame({"Source": read_file(goto_dev)})
        pipeline_test = pd.DataFrame({"Source": read_file(goto_test)})

    if task == 'sr':
        dataset = {
            "pipeline_eval": pd.DataFrame(pipeline_eval),
            "pipeline_test": pd.DataFrame(pipeline_test)
        }

        dataset_dict = DatasetDict({
            "pipeline_eval": Dataset.from_pandas(dataset["pipeline_eval"]),
            "pipeline_test": Dataset.from_pandas(dataset["pipeline_test"])
        })
    else:
        dataset = {
            "train": pd.DataFrame(train_df),
            "validation": pd.DataFrame(dev_df),
            "test": pd.DataFrame(test_df),
            "pipeline_eval": pd.DataFrame(pipeline_eval),
            "pipeline_test": pd.DataFrame(pipeline_test)
        }

        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(dataset["train"]),
            "validation": Dataset.from_pandas(dataset["validation"]),
            "test": Dataset.from_pandas(dataset["test"]),
            "pipeline_eval": Dataset.from_pandas(dataset["pipeline_eval"]),
            "pipeline_test": Dataset.from_pandas(dataset["pipeline_test"])
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
