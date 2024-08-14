import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from transtruct import create_instruction, generate_examples


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


model_id = "CohereForAI/c4ai-command-r-plus-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
pipe = pipeline("text-generation", model, tokenizer = tokenizer)

# Declare variables here
shot = 5
data_path = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/gem/results"
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
    files = [file for file in files if 'gpt' in files]

    # Process each file
    for file_name in files[1:]:#[::-1]:
        print(file_name)
        # Read lines from the file
        file_path = os.path.join(input_folder_path, file_name)
        print(file_path)
        dataset = read_file_lines(file_path)

        print(f"Evaluating {shot}_shot {file_name}.txt dataset:")
        path = os.path.join(write_path, f'{file_name}')

        #Use only one language
        lang = 'sw' #, 'ko', 'ar', 'sw'
        results = []
        for batch_idx, source in enumerate(dataset):
            print(source)
            prompt_in = create_instruction(lang, source)
            messages = [{"role": "user", "content": prompt_in}]
            output = pipe(messages, do_sample=True, max_new_tokens=256)[0]["generated_text"][-1]
            #result = output.replace('\n', '  ')
            #result = output.split(prompt_in)[-1].strip().replace('\n', '  ')
            print(f"{batch_idx} --- {output}\n")
            results.append(output)

        # Write the results into the path
        out_path = f"{path}_{lang}"
        write_file(out_path, results, mode='w')
        print(f'{out_path} Ended!!!!', "\n")
        print(f'{file_name}.txt Ended!!!!', "\n")
