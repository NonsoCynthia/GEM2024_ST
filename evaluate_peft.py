import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, MistralForCausalLM, AutoConfig
#from peft import prepare_model_for_kbit_training, PeftModel, PeftConfig
from load_dataset import CustomDataset, preprocess_data, read_file
from instruction import generate_examples, instruct_templates, write_file


def load_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    #config = AutoConfig.from_pretrained(config_path)
    #config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    #model = PeftModel.from_pretrained( config.base_model_name_or_path, quantization_config=bnb_config, device_map='auto'))
    #model = prepare_model_for_kbit_training(model)

    return model, tokenizer

def generate_text(model, tokenizer, prompt):
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
    #model_id = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/gem/gem_outputs/checkpoint-10000"
    #model_id="mistralai/Mistral-7B-Instruct-v0.2"
    model_id="Mollel/swahili-Instruct-llama-2-7b"
    model, tokenizer = load_model(model_id) 

    system_inst = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being socially unbiased and safe. If you're unsure about an answer, it's okay to skip it, and please ensure not to provide incorrect information. Additionally, responses should be concise and informative."
    text="Alan Shepard was born in New Hampshire and passed away in California. He was awarded the Distinguished Service Medal by the United States Navy."
    instruct = f"""<s>[INST]<<SYS>>Instruction: Translate this text from English to Swahili. Be truthfull, concise and faithfull to the input. Do not omit or add any information not inferred from the text during translation.<</SYS>>\nInput:\n{text}\nOutput:\n[/INST]</s>"""
    generated_texts = generate_text(model, tokenizer, instruct)
    result=''.join(generated_texts).split(instruct)[-1].strip()
    print(result)

    #triples0 = '''[TRIPLE] Adolfo_Su\u00e1rez_Madrid\u2013Barajas_Airport  operatingOrganisation  ENAIRE [/TRIPLE] [TRIPLE] Adolfo_Su\u00e1rez_Madrid\u2013Barajas_Airport  location  Alcobendas [/TRIPLE] [TRIPLE] Adolfo_Su\u00e1rez_Madrid\u2013Barajas_Airport  runwayLength  4349.0 [/TRIPLE] [TRIPLE] Adolfo_Su\u00e1rez_Madrid\u2013Barajas_Airport  runwayName  \"14R/32L\" [/TRIPLE] [TRIPLE] ENAIRE  city  Madrid [/TRIPLE] [TRIPLE] Madrid  country  Spain [/TRIPLE]''' #.replace('[/TRIPLE] [TRIPLE]',', ').replace("[TRIPLE]", '').replace("[/TRIPLE]", '')

    #triples1 = '''[TRIPLE] It's_Great_to_Be_Young_(1956_film)  starring  John_Mills [/TRIPLE] [TRIPLE] It's_Great_to_Be_Young_(1956_film)  runtime  94.0 [/TRIPLE] [TRIPLE] It's_Great_to_Be_Young_(1956_film)  director  Cyril_Frankel [/TRIPLE] [TRIPLE] It's_Great_to_Be_Young_(1956_film)  musicComposer  Louis_Levy [/TRIPLE] [TRIPLE] It's_Great_to_Be_Young_(1956_film)  starring  Cecil_Parker [/TRIPLE]'''.replace("[TRIPLE]", '').replace("[/TRIPLE]", "")

    #triples2 =  "[TRIPLE] Turn_Me_On_(album)  precededBy  Let_It_Breed [/TRIPLE] [TRIPLE] Turn_Me_On_(album)  producer  Wharton_Tiers [/TRIPLE]"
    #triples3 = "[TRIPLE] Est\u00e1dio_Municipal_Coaracy_da_Mata_Fonseca  location  Arapiraca [/TRIPLE] [TRIPLE] Agremia\u00e7\u00e3o_Sportiva_Arapiraquense  league  Campeonato_Brasileiro_S\u00e9rie_C [/TRIPLE] [TRIPLE] Agremia\u00e7\u00e3o_Sportiva_Arapiraquense  season  2015 [/TRIPLE] [TRIPLE] Campeonato_Brasileiro_S\u00e9rie_C  champions  Vila_Nova_Futebol_Clube [/TRIPLE] [TRIPLE] Agremia\u00e7\u00e3o_Sportiva_Arapiraquense  numberOfMembers  17000 [/TRIPLE] [TRIPLE] Agremia\u00e7\u00e3o_Sportiva_Arapiraquense  ground  Est\u00e1dio_Municipal_Coaracy_da_Mata_Fonseca [/TRIPLE] [TRIPLE] Agremia\u00e7\u00e3o_Sportiva_Arapiraquense  fullName  \"Agremia\u00e7\u00e3o Sportiva Arapiraquense\" [/TRIPLE]"

    #instruct = f"""Instruction:\nYou will receive a list of triples containing information about a subject entity. Study the triples carefully and use the information within them to generate a concise text describing key facts about the subject entity. Do not include any information that cannot be directly inferred from the given triples. Your text should be a single coherent paragraph written in proper English.\n\nInput:\n{temple0}\nOutput:\n"""

    #for i, source in enumerate([triples0, triples1, triples2, triples3]):
        #prompt_in = instruct_templates('mistral', '1', source, pipeline=False)
        #prompt_in = prompt_in.replace('[/TRIPLE] [TRIPLE]', ', ').replace('[TRIPLE]','')
        #prompt_in = prompt_in.replace('[/TRIPLE]','').replace('[SNT] [SNT]', '[SNT], [SNT]')

        #generated_texts = generate_text(model, tokenizer, prompt_in)
        #print(i)

    #for text in generated_texts:
        #print(f"Result: {text.split(prompt_in)[-1]}")
        #print(f"Result:{text}")
