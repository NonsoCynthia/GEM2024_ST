#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================================================================
"""
    The module is for translating English text into Hindi text.
    usage: python hi_translate.py
"""
# =================================================================================================
# Imports
# ================================================================================================

import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import argparse
import os

# ================================================================================================

# Before running this code,
# 1. Make sure you have jedi>=0.16, nltk, sacremoses, pandas, regex, mock, transformers>=4.33.2,
#  mosestokenizer, bitsandbytes, scipy, accelerate, datasets, sentencepiece, sacrebleu
# 2. git clone https://github.com/AI4Bharat/IndicTrans2.git
# 3. cd /content/IndicTrans2/huggingface_interface
# 4. git clone https://github.com/VarunGumma/IndicTransTokenizer
# 5. cd IndicTransTokenizer
# 6. python -m pip install --editable ./
# 7. cd ..
# 8. python hi_translate.py

# ================================================================================================

BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
quantization = None

def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()

    return tokenizer, model

def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations

def translate_file(input_path, output_path, src_lang, tgt_lang, ckpt_dir, quantization):
    with open(input_path, 'r') as f_open:
        en_sents = [line for line in f_open.readlines()]

    f_out = open(output_path, 'w')

    tokenizer, model = initialize_model_and_tokenizer(ckpt_dir, f"{src_lang}-indic", quantization)

    ip = IndicProcessor(inference=True)

    hi_translations = batch_translate(en_sents, src_lang, tgt_lang, model, tokenizer, ip)

    print(f"\n{src_lang} - {tgt_lang}")
    for input_sentence, translation in zip(en_sents, hi_translations):
        print(f"{src_lang}: {input_sentence}")
        print(f"{tgt_lang}: {translation}")
        print()
        f_out.write(translation+'\n')

    # Flush the models to free the GPU memory
    del tokenizer, model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Translate generated English Text to target languages for GEM 2024 Shared Task test dataset")
    parser.add_argument("--project_dir", help="path to the input directory", default = os.path.join('/', 'home', 'your project path'))
    parser.add_argument("--out_dir", help="path to the output directory", default = os.path.join('/', 'home', 'your project path'))
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    project_dir_path = args.project_dir
    out_dir = args.out_dir
    tgt_lang = 'hi'


    for root, _, filenames in os.walk(project_dir_path):
        for filename in filenames:
            in_file = os.path.join(root, filename)
            out_file = os.path.join(out_dir, f"{tgt_lang}_{filename}")
            translate_file(
            input_path=in_file,
            output_path=out_file,
            src_lang="eng_Latn",
            tgt_lang="hin_Deva",
            ckpt_dir="ai4bharat/indictrans2-en-indic-1B",
            quantization=None
            )
