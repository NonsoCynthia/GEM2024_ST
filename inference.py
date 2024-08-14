import os
import json
import torch
import argparse
#import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict, Dataset
from load_dataset import  preprocess_data_
from data.load_dataset import CustomDataset, preprocess_data, realize_date
from T5 import T5_Model
from torchmetrics.classification import StatScores
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from mapping import entity_mapping, prcs_entry, delist, split_triples

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_input):
        self.data_input = data_input

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, idx):
        return self.data_input[idx]

class Inferencer:
    def __init__(self, model, testdata, task, batch_status, write_dir, verbose=True):
        self.model = model
        self.testdata = testdata
        self.task = task
        self.batch_status = batch_status
        self.write_dir = write_dir
        self.verbose = verbose

        if not os.path.exists(write_dir):
            os.mkdir(write_dir)

    def evaluate(self):
        # Evaluate the model's performance
        self.model.model.eval()  # sets the model in evaluation mode using eval function
        results = []
        for batch_idx, inputs in enumerate(self.testdata):
            source = inputs.get('Source', None)
            targets = inputs.get('Target', None)

            # if source not in results:
            # Initialize the dictionary for this source
            result = {'idx': batch_idx, 'input': source, 'pred': '',
                      'refs': []}  # hyp=generated texts; refs=original targets

            # Predict
            output = self.model([source])
            result['pred'] = output[0].strip()

            # Display evaluation progress
            if (batch_idx + 1) % self.batch_status == 0:
                print('Evaluation: [{}/{} ({:.0f}%)]'.format(batch_idx + 1, \
                                                             len(self.testdata), 100. * batch_idx / len(self.testdata)))

            # Store references as a list of lists
            result['refs'].append(targets)

            # Add the result to the results list
            results.append(result)

        # Sort results
        results = sorted(results, key=lambda x: x['idx'])
        # Define result directory path
        path = os.path.join(self.write_dir, f'{self.task}.txt')
        with open(path, 'w') as f:
            f.write('\n'.join([w['pred'] for w in results]))

        path = os.path.join(self.write_dir, f'{self.task}.json')
        json.dump(results, open(path, 'w'), separators=(',', ':'), sort_keys=True, indent=4)

        if targets is not None:
            hyps, refs = [], []
            for i, row in enumerate(results):
                hyps.append(nltk.word_tokenize(row['pred']))
                refs.append([nltk.word_tokenize(ref) for ref in row['refs']])

            chencherry = SmoothingFunction()
            bleu = corpus_bleu(refs, hyps, smoothing_function=chencherry.method3)
            print(f'BLEU: {bleu}')

            return bleu


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", help="path to the tokenizer")
    parser.add_argument("--model_path", help="path to the model")
    parser.add_argument("--task", help="Traing task")
    parser.add_argument("--data_path", help="path to the data")
    parser.add_argument("--batch_size", help="batch size of test", type=int)
    parser.add_argument("--max_length", help="maximum length to be processed by the network", type=int)
    parser.add_argument("--verbose", help="should display the loss?", action="store_true")
    parser.add_argument("--batch_status", help="display of loss", type=int)
    # parser.add_argument("--cuda", help="use CUDA", action="store_true")
    args = parser.parse_args()

    # Model settings, Settings and configurations
    tokenizer_path = args.tokenizer
    model_path = args.model_path
    task = args.task
    data_path = args.data_path
    batch_size = args.batch_size
    max_length = args.max_length
    verbose = args.verbose if 'verbose' in args else False
    batch_status = args.batch_status
    # device = torch.device('cuda' if args.cuda else 'cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_path = os.path.join(args.model_path, args.task)

    # Create model
    # /home/cosuji/spinning-storage/cosuji/NLG_Exp/webnlg/results/ordering/model
    model_path = os.path.join(model_path, f"{task}/model")
    generator = T5_Model(tokenizer_path, model_path, max_length, sep_token=task + ":")


    dataset_dict = preprocess_data(data_path, task, mod)
    evaluation = {
        f"{task}_dev": dataset_dict["validation"],
        f"{task}_test": dataset_dict["test"],
        f"{task}_pipeline_eval": dataset_dict["pipeline_eval"],
        f"{task}_pipeline_test": dataset_dict["pipeline_test"],  # Note: Assuming this is intentional
    }

    for dataset_name, dataset in evaluation.items():
        inf = Inferencer(generator, dataset, dataset_name, batch_status, write_path, verbose)
        print(f"Evaluating {dataset_name} dataset:")
        inf.evaluate()
