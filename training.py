import os
# Set the environment variable to disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import nltk
import torch
import torch.nn as nn
from torch import optim
#import pytorch_lightning as pl
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
nltk.download('punkt')
#import wandb
from transformers import set_seed
set_seed(42)

# Create a Trainer class based on PyTorch Lightning's LightningModule
class Trainer(nn.Module):
    def __init__(self, model, trainloader, devdata, optimizer, epochs, batch_status, write_path, early_stop=5, verbose=True):
        # Initialize the Trainer class
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_status = batch_status
        self.early_stop = early_stop
        self.verbose = verbose
        self.trainloader = trainloader
        self.devdata = devdata
        self.write_path = write_path
        
        if not os.path.exists(write_path):
            os.mkdir(write_path)

        # Initialize WandB
        # wandb_project = 'webnlg'  # args.wandb_project
        # wandb_entity = 'afrisent-nlp'  # args.wandb_entity
        # wandb.init(project=wandb_project, entity=wandb_entity)

    def train(self):
        # Train the model
        max_bleu, repeat = 0, 0
        for epoch in range(self.epochs):
            # For the given number of epochs train the number
            self.model.model.train() #sets the model in evaluation mode using train function
            losses = []
            for batch_idx, inputs in enumerate(self.trainloader):
                source, targets = inputs['Source'], inputs['Target']

                #if isinstance(inputs, (list, tuple)):
                    #source, targets = inputs[0], inputs[1]  # Assuming inputs is a list/tuple of (source, targets)
                #else:
                    #raise TypeError("Unexpected input format. Expected list or tuple.")

                self.optimizer.zero_grad()

                # generating
                output = self.model(source, targets)

                # Calculate loss
                loss = output.loss
                losses.append(float(loss))

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                # Log loss to WandB
                # wandb.log({'train_loss': loss})

                # Display training progress
                if (batch_idx + 1) % self.batch_status == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTotal Loss: {:.6f}'.format(epoch, \
                            batch_idx + 1,len(self.trainloader),100. * batch_idx / len(self.trainloader), \
                            float(loss),round(sum(losses) / len(losses),5)))

            bleu = self.evaluate()
            print("Model: ", '-'.join(str(self.write_path).split('/')[-2:]), 'BLEU: ', bleu)
            # wandb.log({'bleu_score': bleu})

            if bleu > max_bleu:
                self.model.model.save_pretrained(os.path.join(self.write_path, 'model'))
                # wandb.save(self.write_path)
                max_bleu = bleu
                repeat = 0
                print('Saving best model...')
            else:
                repeat += 1

            if repeat == self.early_stop:
                break

    def evaluate(self):
        # Evaluate the model's performance
        self.model.model.eval() #sets the model in evaluation mode using eval function
        results = {}
        for batch_idx, inputs in enumerate(self.devdata):
            source, targets = inputs['Source'], inputs['Target']
            if source not in results:
                # Initialize the dictionary for this source
                # hyp is the generated texts; refs is the original targets
                results[source] = {'hyp': '', 'refs': []}

                # Predict
                output = self.model([source])
                results[source]['hyp'] = output[0]

                # Display evaluation progress
                if (batch_idx + 1) % self.batch_status == 0:
                    print('Evaluation: [{}/{} ({:.0f}%)]'.format(batch_idx + 1, \
                                len(self.devdata), 100. * batch_idx / len(self.devdata)))

            # Store references as a list of lists
            results[source]['refs'].append(targets)

        hypothesis, references = [], []
        
        for source in results.keys():
            if self.verbose:
                print('Source:', source)
                for ref in results[source]['refs']:
                    print('Real: ', ref)
                print('Pred: ', results[source]['hyp'])
                print()

            # Tokenize hypotheses and references
            hypothesis.append(nltk.word_tokenize(results[source]['hyp']))
            references.append([nltk.word_tokenize(ref) for ref in results[source]['refs']])

        chencherry = SmoothingFunction()
        bleu = corpus_bleu(references, hypothesis, smoothing_function=chencherry.method3)
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
    if "t5" in tokenizer_path.lower():
        mod = 't5'
    elif "ul2" in tokenizer_path.lower():
        mod = 'ul2'
    else:
        raise Exception("Invalid model")

    if "gem_data" in data:
        dataset_dict = preprocess_data_(data, task)
        train_dataset = CustomDataset(dataset_dict["train"])
        validation_dataset = dataset_dict["validation"]
        inference_test = dataset_dict["inference_test"]
    else:
        dataset_dict = preprocess_data(data, task, mod)
        train_dataset = CustomDataset(dataset_dict["train"])
        validation_dataset = dataset_dict["validation"]
        test_dataset = dataset_dict["test"]
    
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
