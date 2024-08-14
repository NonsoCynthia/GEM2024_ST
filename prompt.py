import os
import re
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import DataLoader
from load_dataset import CustomDataset, preprocess_data, read_file
from dotenv import load_dotenv, find_dotenv
from instruction import generate_examples, instruct_templates, write_file
#import hf_olmo

_ = load_dotenv(find_dotenv())  # read local .env file
hf_token = os.getenv("HF_TOKEN")


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
        #if "gem_outputs" in model_id:
            #config_path = '/home/cosuji/spinning-storage/cosuji/NLG_Exp/gem/gem_outputs'
            #model.config.use_cache = False

        return model

    def forward(self, source, targets=None):
        input_ids = self.tokenizer.encode(source, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(input_ids.to(torch.long), pad_token_id=self.tokenizer.eos_token_id, do_sample=True, 
                                                 max_new_tokens=512, num_beams=2, repetition_penalty=2.0)
            prompt_len = input_ids.shape[-1] #[prompt_len:]
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()      
            generated_text = re.sub('\n+', '\n', generated_text)  # remove excessive newline characters
        return generated_text

class Inferencer:
    def __init__(self, model, testdata, shot, model_name, write_dir):
        self.model = model
        self.testdata = testdata
        self.shot = shot
        self.model_name = model_name
        self.write_dir = write_dir


    def evaluate(self):
        self.model.eval()
        results = []
        for batch_idx, inputs in enumerate(self.testdata):
            source = inputs.get('Source', None)
            if source:
                # Predict
                if "struct" in self.write_dir:
                    prompt_in = instruct_templates(self.model_name, self.shot, source, pipeline=True)
                else:
                    prompt_in = instruct_templates(self.model_name, self.shot, source, pipeline=False)
                
                prompt_in = prompt_in.replace('[/TRIPLE] [TRIPLE]', ', ').replace('[TRIPLE]','')
                prompt_in = prompt_in.replace('[/TRIPLE]','').replace('[SNT] [SNT]', '[SNT], [SNT]')
                output = self.model(prompt_in)
                result = output.split(prompt_in)[-1].strip().replace('\n', '  ')
                print(batch_idx,'  ',result)
                results.append(result)
            else:
                print("No Source")

        # Write the results into the path
        write_file(self.write_dir, results, mode='w')
        print(f'{self.write_dir} Ended!!!!', "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to the model in hugging face or local path")
    parser.add_argument("--model_type", help="Type of the model base or finetuned")
    parser.add_argument("--model_name", help="Shortened name of the models")
    parser.add_argument("--shot", help="zero/few shot training task")
    parser.add_argument("--data_path", help="path to the dataset")
    parser.add_argument("--write_path", help="path to write the results")
    args = parser.parse_args()

    model_id = args.model
    model_type = args.model_type
    model_name = args.model_name
    shot = args.shot
    data_path = args.data_path
    write_path = args.write_path

    write_path = os.path.join(write_path, f'{shot}_shot')
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    if "gem_data" in data_path or "gem_results" in data_path:
        dataset_dict = preprocess_data(data_path, task=None)
        evaluation = {}
        for key in dataset_dict.keys():
            #if 'struct' in key:
            evaluation[key] = dataset_dict[key]
    else:
        raise ValueError("The path provided is not for inference")

    hf_auth = hf_token
    max_length = 1024
    llama_model = LlamaModel(model_id, hf_auth, max_length)

    for dataset_name, dataset in evaluation.items():
        print(f"Evaluating {shot}_shot {model_name} {model_type} {dataset_name}.txt dataset:")
        path = os.path.join(write_path, f'{dataset_name}_{model_name}_{model_type}_new.txt')
        inf = Inferencer(llama_model, dataset, shot, model_name, path)
        inf.evaluate()
        print(f'{dataset_name}.txt Ended!!!!', "\n")
