import os
import re
import json
import torch
import argparse
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, MistralForCausalLM, AutoConfig, set_seed
from peft import prepare_model_for_kbit_training, PeftModel, PeftConfig
from load_dataset import CustomDataset, preprocess_data, read_file
from instruction import generate_examples, instruct_templates, write_file
set_seed(42)

def load_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    #config = AutoConfig.from_pretrained(model_name)
    #config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    config = PeftConfig.from_pretrained(model_name)

    #model = AutoModelForCausalLM.from_pretrained(
        #model_name,
        #config=config,
        #config.base_model_name_or_path,
        #quantization_config=bnb_config,
        #trust_remote_code=True,
        #low_cpu_mem_usage=True,
        #device_map="auto",
    #)
    device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map=device_map, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, model_name)
    #model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                             trust_remote_code=True,
                                             padding=True, 
                                             truncation=True, 
                                             add_eos_token=False
                                             )
    tokenizer.pad_token = tokenizer.eos_token

    #model = PeftModel.from_pretrained( config.base_model_name_or_path, quantization_config=bnb_config, device_map='auto'))
    #model = prepare_model_for_kbit_training(model)

    return model, tokenizer

def generate_text(model, tokenizer, prompt):
    # Set pad_token_id to None to avoid the warning
    #tokenizer.pad_token_id = None
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    sequences = pipeline(
        prompt,
        max_new_tokens=300,
        do_sample=True,
        #top_k=10,
        #num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = len(prompt)
    generated_texts = [seq['generated_text'] for seq in sequences]
    return generated_texts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path to the model in hugging face or local path")
    parser.add_argument("--model_type",type=str,  help="Type of the model base or finetuned")
    parser.add_argument("--model_name", type=str,  help="Shortened name of the models")
    parser.add_argument("--shot", help="zero/few shot training task")
    parser.add_argument("--data_path",type=str, help="path to the dataset")
    parser.add_argument("--write_path", type=str, help="path to write the results")
    args = parser.parse_args()

    model_id = args.model
    model_type = args.model_type
    model_name = args.model_name
    shot = args.shot
    data_path = args.data_path
    write_path = args.write_path
    #model_id = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/gem/gem_outputs/checkpoint-10000"
    model, tokenizer = load_model(model_id) 

    write_path = os.path.join(write_path, f'{shot}_shot')
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    if "gem_data" in data_path or "gem_results" in data_path:
        dataset_dict = preprocess_data(data_path, task=None)
        evaluation = {}
        for key in dataset_dict.keys():
            if 'struct' in key:
                evaluation[key] = dataset_dict[key]
    else:
        raise ValueError("The path provided is not for inference")

    for dataset_name, dataset in evaluation.items():
        print(f"Evaluating {shot}_shot {model_name} {model_type} {dataset_name}.txt dataset:")
        path = os.path.join(write_path, f'{dataset_name}_{model_name}_{model_type}_new.txt')
        results = []
        for batch_idx, inputs in enumerate(dataset):
            source = inputs.get('Source', None)
            if source:
                # Predict
                if "struct" in write_path:
                    prompt_in = instruct_templates(model_name, shot, source, pipeline=True)
                else:
                    prompt_in = instruct_templates(model_name, shot, source, pipeline=False)

                prompt_in = prompt_in.replace('[/TRIPLE] [TRIPLE]', ', ').replace('[TRIPLE]','')
                prompt_in = prompt_in.replace('[/TRIPLE]','').replace('[SNT] [SNT]', '[SNT], [SNT]')
                generated_texts = generate_text(model, tokenizer, prompt_in)
                output = ''.join(generated_texts)
                result = output.strip().replace('\n', '  ')
                print(batch_idx,'  ',result)
                results.append(result)
             #else:
                #print("No Source")

        # Write the results into the path
        write_file(path, results, mode='w')
        print(f'{path} Ended!!!!', "\n")
        print(f'{dataset_name}.txt Ended!!!!', "\n")
