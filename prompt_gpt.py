import os
import time
import argparse
from dotenv import load_dotenv, find_dotenv
from data.load_dataset import CustomDataset, preprocess_data
from openai import OpenAI
from load_dataset import CustomDataset, preprocess_data, read_file
from instruction import generate_examples, instruct_templates, write_file

_ = load_dotenv(find_dotenv())  # read local .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_completion(prompt, model):  # model="gpt-3.5-turbo", 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo', 'gpt-4'
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()

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

    write_path = os.path.join(write_path, f'{shot}_shot')
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    if "gem_data" in data_path or "gem_results" in data_path:
        dataset_dict = preprocess_data(data_path, task=None)
        evaluation = {}
        for key in dataset_dict.keys():
            if "struct" in key:
                evaluation[key] = dataset_dict[key]
    else:
        raise ValueError("The path provided is not for inference")

    for dataset_name, dataset in evaluation.items():
        print(f"Evaluating {shot}_shot {model_name} {model_type} {dataset_name}.txt dataset:")
        path = os.path.join(write_path, f'{dataset_name}_{model_name}_{model_type}.txt')
        results = []
        for batch_idx, inputs in enumerate(dataset):
            source = inputs.get('Source', None)
            if source:
                # Predict
                if "struct" in write_path:
                    prompt_in = instruct_templates(model_name, shot, source, pipeline=True)
                else:
                    prompt_in = instruct_templates(model_name, shot, source, pipeline=False)

                generated_texts = get_completion(prompt_in, model_id)
                result = generated_texts.strip().replace('Output: ','').replace('\n', '  ')
                print(batch_idx,'  ',result)
                results.append(result)

        # Write the results into the path
        write_file(path, results, mode='w')
        print(f'{path} Ended!!!!', "\n")
        print(f'{dataset_name}.txt Ended!!!!', "\n")
