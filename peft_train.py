import os
import sys
import json
import random
from random import randrange
from random import randint
import random
from sklearn.model_selection import train_test_split
import torch
import wandb
from datasets import Dataset, load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments)
from templates import load_json_data, templates, inverse_templates



def prepare_instructions(webnlg, tokenizer, templates, inverse_templates):
    instruction_data = []
    inverse_instruction_data = []
    delims = ['|', '###', ',', ';']
    swap = False
    counter = 0
    for inc, web in enumerate(webnlg):
        order_in = '\n'.join(['  '.join(chunk.split()) for chunk in web["order_in"]])
        if swap:
            delim = random.choice(delims)
            order_in = '\n'.join([f' {delim} '.join(chunk.split()) for chunk in web["order_in"]])
            swap = False
            counter = 0
        text = web["text"]
        template = random.choice(templates)
        instruction_data.append(template.format(order_in=order_in, text=text, tokenizer=tokenizer))
        template2 = random.choice(inverse_templates)
        inverse_instruction_data.append(template2.format(order_in=order_in, text=text, tokenizer=tokenizer))
        counter += 1
    random.seed(42)  # Setting the seed for reproducibility
    random.shuffle(instruction_data)  # Shuffle the instruction_data list

    tr_size = int(len(instruction_data) * 0.8)  # Convert tr_size to an integer
    train_dataset = instruction_data[:tr_size]
    dev_dataset = instruction_data[tr_size:]
    return train_dataset, dev_dataset
    #return instruction_data #+ inverse_instruction_data


def create_dataset(data, tokenizer):
    dataset = Dataset.from_dict({"text": data})
    return dataset.map(lambda sample: tokenizer(sample["text"]), batched=True)

def configure_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, 
                                    bnb_4bit_use_double_quant=True, 
                                    bnb_4bit_quant_type="nf4", 
                                    bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config)

    return model, tokenizer

def configure_training(model, tokenizer, train_dataset, dev_dataset, peft_config, args, max_seq_length):
    trainer = SFTTrainer(model=model,
                         args=args,
                         train_dataset=train_dataset,
                         eval_dataset=dev_dataset,
                         peft_config=peft_config,
                         max_seq_length=max_seq_length,
                         tokenizer=tokenizer,
                         dataset_text_field="text",
                         packing=True,
                         dataset_kwargs={
                             "add_special_tokens": False,
                             "append_concat_token": False,
                         })
    return trainer

if __name__ == "__main__":
    webnlg_path ='/home/cosuji/spinning-storage/cosuji/NLG_Exp/gem/gem_data/webnlg_17_data.json'
    webnlg = load_json_data(webnlg_path)
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model, tokenizer = configure_model_and_tokenizer(model_id)
    
    instructions = prepare_instructions(webnlg, tokenizer, templates, inverse_templates)
    train_dataset, dev_dataset  = prepare_instructions(webnlg, tokenizer, templates, inverse_templates)    
    #dataset = Dataset.from_dict({"text": instructions})
    #print(dataset[randint(0, len(dataset))]["text"])

    #train_dataset, dev_dataset = create_dataset(instructions, tokenizer)
    train_dataset = create_dataset(train_dataset, tokenizer)
    dev_dataset = create_dataset(dev_dataset, tokenizer)

    if 'Mistral' in model_id:
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ]
    else:
        target_modules=[
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ]

    peft_config = LoraConfig(lora_alpha=128,
                             lora_dropout=0.1,
                             r=256,
                             bias="none",
                             target_modules=target_modules,
                             task_type="CAUSAL_LM")
    
    args = TrainingArguments(output_dir="../gem_outputs",
                             num_train_epochs=4,
                             per_device_train_batch_size=4,
                             per_device_eval_batch_size=1,
                             gradient_accumulation_steps=4,
                             gradient_checkpointing=True,
                             optim="adamw_torch_fused",
                             logging_steps=50,
                             save_strategy="epoch",
                             learning_rate=2e-4,
                             bf16=True,
                             tf32=True,
                             max_grad_norm=0.3,
                             warmup_ratio=0.03,
                             lr_scheduler_type="constant",
                             max_steps=10000,
                             load_best_model_at_end=True,
                             overwrite_output_dir=True,
                             evaluation_strategy="epoch",
                             push_to_hub=False,
                             report_to='wandb',
                             run_name='GEM webnlg finetuning')

    max_seq_length = 512

    trainer = configure_training(model, tokenizer, train_dataset, dev_dataset, peft_config, args, max_seq_length)
    trainer.train()
    trainer.save_model()

