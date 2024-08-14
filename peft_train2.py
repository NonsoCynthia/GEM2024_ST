import copy
import torch
from transformers import pipeline
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import Dataset

def load_model_and_tokenizer(model_id, use_peft=False):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16)

    if use_peft:
        model = AutoPeftModelForCausalLM.from_pretrained(model_id,
                                                         torch_dtype=torch.float16,
                                                         low_cpu_mem_usage=True)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(model_id, safe_serialization=True, max_shard_size="2GB")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     device_map="auto",
                                                     torch_dtype=torch.bfloat16,
                                                     quantization_config=bnb_config)

    return model, tokenizer

# Assuming 'webnlg' is defined elsewhere
def prepare_data(webnlg):
    reverse_web = []
    for inc, item in enumerate(webnlg):
        item_ = copy.deepcopy(item)
        item_['order_in'] = '\n'.join(['  '.join(chunk.split()) for chunk in item_["order_in"]])
        item['key'] = 0
        reverse_web.append(item_)

    reverse_web_ = []
    for item in reverse_web:
        item['order_in'], item['text'] = item['text'], item['order_in']
        item['key'] = 1
        reverse_web_.append(item)

    webnlg2 = reverse_web + reverse_web_

    text_data = [data['text'] for data in webnlg2]
    order_data = [data['order_in'] for data in webnlg2]
    key_data = [data['key'] for data in webnlg2]

    dataset = Dataset.from_dict({"text": text_data, "order-in": order_data, "key": key_data})

    return dataset

def format_prompts(examples, tokenizer, templates):
    output_texts = []
    for example in examples:
        order_in = example["order_in"]
        text = example["text"]
        if example['key'] == 0:
            prompt = templates[0].format(order_in=order_in, text=text, tokenizer=tokenizer)
        else:
            prompt = templates[1].format(order_in=order_in, text=text, tokenizer=tokenizer)
        output_texts.append(prompt)
    return output_texts

bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16)

# Load the appropriate model based on the model_id
def load_model(model_id):
    use_peft = 'Peft' in model_id
    model, tokenizer = load_model_and_tokenizer(model_id, use_peft=use_peft)
    return model, tokenizer




if __name__ == "__main__":
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    model, tokenizer = load_model(model_id)

    # Prepare data
    dataset = prepare_data(webnlg)

    # Define templates for formatting prompts
    template_0 = """Instruction:\nGenerate factual and coherent text from the following input. \n\nInput:\n{order_in}\n\nOutput:\n{text}{tokenizer.eos_token}"""
    template_1 = """Instruction:\nGenerate relevant and coherent triples indicating the entity and associated attribute relationships from the given text. \n\nInput:\n{text}\n\nOutput:\n{order_in}{tokenizer.eos_token}"""
    templates = [template_0, template_1]

    # Format prompts
    formatted_prompts = format_prompts(dataset, tokenizer, templates)

    args = TrainingArguments(
        output_dir="gem_output",
        num_train_epochs=3,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=2,
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
        push_to_hub=False,
        report_to='wandb',
        run_name='GEM webnlg finetuning'
    )

    collator = DataCollatorForCompletionOnlyLM('\n\nOutput:', tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        formatting_func=format_prompts,
        data_collator=collator,
        peft_config=peft_config if 'Peft' in model_id else None,
        max_seq_length=3072,
        tokenizer=tokenizer
    )

    trainer.train()

