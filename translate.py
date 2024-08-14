import os
import re
import torch
import argparse
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader
from load_dataset import CustomDataset, preprocess_data, read_file
from dotenv import load_dotenv, find_dotenv
from transtruct import create_instruction, generate_examples
#import hf_olmo

_ = load_dotenv(find_dotenv())  # read local .env file
hf_token = os.getenv("HF_TOKEN")


def read_file_lines(path):
    with open(path, 'r', encoding='utf-8') as file:
        contents = file.read()
        lines = [line.strip() for line in contents.split('\n')]
        if lines and lines[-1] == '':
            return lines[:-1]
        return lines

# Function to write files
def write_file(write_path, result, mode='w'):
    with open(write_path, mode) as f:
        f.write('\n'.join(result))

class LlamaModel(torch.nn.Module):
    def __init__(self, model_id, hf_auth, max_length=512):
        super(LlamaModel, self).__init__()
        self.max_length = max_length
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_auth) #, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side="left" if 'SOLAR' in model_id else 'right'  
        self.model = self.load_model(model_id, hf_auth)

    def load_model(self, model_id, hf_auth):
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_quant_type='nf4',
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_compute_dtype=torch.bfloat16
                                        )
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     quantization_config=bnb_config,
                                                     trust_remote_code=True,
                                                     token=hf_auth,
                                                     #use_safetensors=True,
                                                     device_map="auto",
                                                     low_cpu_mem_usage=True, 
                                                     )

        return model

    def forward(self, source, targets=None):
        # Format message with the command-r chat template
        messages = [{"role": "user", "content": source}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        ###input_ids = self.tokenizer.encode(source, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(input_ids.to(torch.long), pad_token_id=self.tokenizer.eos_token_id, do_sample=True, 
                                                 max_new_tokens=300, num_beams=2, repetition_penalty=2.0)
            #prompt_len = input_ids.shape[-1] #[prompt_len:]
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()      
            generated_text = re.sub('\n+', '\n', generated_text)  # remove excessive newline characters
        
        #Use transformer pipelines
        #pipeline = transformers.pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, 
                                                        #torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        #source = f'<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{source}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'
        #sequences = pipeline(source, max_new_tokens=300, do_sample=True, repetition_penalty=2.0, num_beams=2, 
                                                                                    #eos_token_id=self.tokenizer.eos_token_id)
        #prompt_len = len(source) [prompt_len:]
        #generated_text = [seq['generated_text'] for seq in sequences]
        #generated_text = ''.join(generated_text)
        #generated_text = re.sub('\n+', '\n', generated_text)

        return generated_text

class Inferencer:
    def __init__(self, model, testdata, shot, write_dir):
        self.model = model
        self.testdata = testdata
        self.shot = shot
        self.write_dir = write_dir


    def evaluate(self):
        self.model.eval()
        #for lang in ['sw', 'ko', 'ar']:
            #results = []
            #for batch_idx, source in enumerate(self.testdata):
                #prompt_in = create_instruction(lang, source)
                #output = self.model(prompt_in)
                #result = output.split(prompt_in)[-1].strip().replace('\n', '  ')
                #print(f"{batch_idx} --- {result}\n")
                #results.append(result)
            # Write the results into the path
            #out_path = f"{self.write_dir}_{lang}"
            #write_file(out_path, results, mode='w')
            #print(f'{out_path} Ended!!!!', "\n")

        #Use only one language
        lang = 'sw' #, 'ko', 'ar', 'sw'
        results = []
        for batch_idx, source in enumerate(self.testdata):
            prompt_in = create_instruction(lang, source)
            output = self.model(prompt_in)
            result = output.split(prompt_in)[-1].strip().replace('\n', '  ')
            print(f"{batch_idx} --- {result}\n")
            results.append(result)
        # Write the results into the path
        out_path = f"{self.write_dir}_{lang}"
        write_file(out_path, results, mode='w')
        print(f'{out_path} Ended!!!!', "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to the model in hugging face or local path")
    parser.add_argument("--shot", help="zero/few shot training task")
    parser.add_argument("--data_path", help="path to the dataset")
    args = parser.parse_args()

    # Declare variables here
    model_id = args.model
    shot = args.shot
    data_path = args.data_path

    hf_auth = hf_token
    max_length = 1024
    llama_model = LlamaModel(model_id, hf_auth, max_length)

    write_path = os.path.join(data_path, 'translate', f'{shot}_shot')
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    print(write_path)

    # Define the input and output folders
    input_folder_path = os.path.join(data_path, 'cleaned', f"{shot}_shot")
    if os.path.exists(input_folder_path):
        files = os.listdir(input_folder_path)
        # Filter files to consider only text files
        files = [file for file in files if file.endswith('.txt')]

        # Process each file
        for file_name in files[1:]:#[::-1]:  
            print(file_name)
            # Read lines from the file
            file_path = os.path.join(input_folder_path, file_name)
            print(file_path)
            dataset = read_file_lines(file_path) 
            #dataset = pd.DataFrame(dataset)
            #print(dataset.head())

            print(f"Evaluating {shot}_shot {file_name}.txt dataset:")
            path = os.path.join(write_path, f'{file_name}')
            inf = Inferencer(llama_model, dataset, shot, path)
            inf.evaluate()
            print(f'{file_name}.txt Ended!!!!', "\n")
