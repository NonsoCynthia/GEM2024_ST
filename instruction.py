

data_examples = {
    1: {
        'Input': 'Uruguay leader Tabaré_Vázquez, Uruguay leader Raúl_Fernando_Sendic_Rodríguez, Alfredo_Zitarrosa deathPlace Montevideo, Montevideo country Uruguay',
        'Output': 'Alfredo Zitarrosa died in Montevideo, Uruguay which is led by Raúl Fernando Sendic Rodríguez and Tabaré Vázquez.'
    },
    2: {
        'Input': 'Angola_International_Airport location Ícolo_e_Bengo, Ícolo_e_Bengo country Angola, Angola_International_Airport cityServed Luanda, Ícolo_e_Bengo isPartOf Luanda_Province, Angola_International_Airport elevationAboveTheSeaLevelInMetres 159',
        'Output': 'Angola International Airport is located at Ícolo e Bengo in Luanda province, Angola. The Airport is situated 159 meters above sea level and serves the city of Luanda.'
    },
    3: {
        'Input': 'United_Petrotrin_F.C. ground Palo_Seco, Akeem_Adams club Trinidad_and_Tobago_national_under-20_football_team, Akeem_Adams club United_Petrotrin_F.C.',
        'Output': 'Akeem Adams, who plays for the Trinidad and Tobago national under-20 football team previously played for United Petrotrin FC whose ground is at Palo Seco.'
    },
    4: {
        'Input': 'William_Anders selectedByNasa 1963, William_Anders nationality United_States, William_Anders birthDate "1933-10-17", William_Anders occupation Fighter_pilot, William_Anders birthPlace British_Hong_Kong, William_Anders mission Apollo_8',
        'Output': 'The United States fighter pilot William Anders was born in British Hong Kong on the 17th of October, 1933. In 1963, he was chosen by NASA and became a crew member on Apollo 8.'
    },
    5: {
        'Input': "Dead_Man's_Plack location England, England ethnicGroup British_Arabs, England capital London, Dead_Man's_Plack dedicatedTo Æthelwald,_Ealdorman_of_East_Anglia, England language Cornish_language, England religion Church_of_England, Dead_Man's_Plack material Rock_(geology)",
        'Output': "The capital of England is London where we can find the Dead Man's Plack which is made of stone. The Plack is dedicated to Æthelwald, Ealdorman of East Anglia. Cornish language is spoken in England and it has an established religion called the Church of England. One of the ethnic groups found in that country is the British Arabs."
    }
}

f='''data_examples_struct = {
    1: {
        "Input": "[SNT] Atatürk_Monument_(İzmir) material 'Bronze', Atatürk_Monument_(İzmir) inaugurationDate '1932-07-27' [/SNT] [SNT] Atatürk_Monument_(İzmir) location Turkey, Turkey capital Ankara, Turkey largestCity Istanbul [/SNT] [SNT] Turkey leaderName Ahmet_Davutoğlu, Turkey currency Turkish_lira [/SNT]",
        #"Output": "The Atatürk Monument is a bronze monument inaugurated on 27th July, 1932, in Izmir. It is found in Turkey, a country which has Ankara as its capital and Istanbul as its largest city. The leader of Turkey is called Ahmet Davutoğlu, and the currency is the Turkish lira."
    },
    2: {
        "Input": "[SNT] Turkey capital Ankara, Turkey largestCity Istanbul [/SNT] [SNT] Turkey leader Ahmet_Davutoğlu, Turkey currency Turkish_lira [/SNT] [SNT] Atatürk_Monument_(İzmir) location Turkey [/SNT]",
        "Output": "The capital of Turkey is Ankara, although the largest city is Istanbul. The leader of Turkey is Ahmet Davutoglu and the currency is known as the Turkish lira. The Ataturk monument is located within the country."
    },
    3: {
        "Input": "[SNT] Antwerp_International_Airport cityServed Antwerp, Antwerp country Belgium, Belgium leaderName Philippe_of_Belgium, Belgium language French_language [/SNT]",
        "Output": "Antwerp is served by Antwerp International Airport and is a popular tourism destination in Belgium where the leader is Philippe of Belgium and the French language is spoken."
    },
    4: {
        "Input": "[SNT] AWH_Engineering_College state Kerala, AWH_Engineering_College country India, AWH_Engineering_College established 2001 [/SNT] [SNT] India river Ganges, India largestCity Mumbai [/SNT] [SNT] Kerala leaderName Kochi [/SNT]",
        "Output": "The AWH Engineering College in Kerala, India was established in 2001. The Ganges is a river in India and its largest city is Mumbai. The leader of Kerala is Kochi."
    },
    5: {
        "Input": "[SNT] Atlanta country United_States, United_States capital Washington_D.C. [/SNT] [SNT] D.C. United_States ethnicGroup Asian_Americans [/SNT]",
        "Output": "Atlanta is in the United States whose capital is Washington, D.C. Asian Americans are an ethnic group in the U.S."
    }
}'''

data_examples_struct = {
    1: {
        "Input": "[SNT] [TRIPLE] Atatürk_Monument_(İzmir) material 'Bronze' [/TRIPLE] [TRIPLE] Atatürk_Monument_(İzmir) inaugurationDate '1932-07-27' [/TRIPLE] [/SNT] [SNT] [TRIPLE] Atatürk_Monument_(İzmir) location Turkey [/TRIPLE] [TRIPLE] Turkey capital Ankara [/TRIPLE] [TRIPLE] Turkey largestCity Istanbul [/TRIPLE] [/SNT] [SNT] [TRIPLE] Turkey leaderName Ahmet_Davutoğlu [/TRIPLE] [TRIPLE] Turkey currency Turkish_lira [/TRIPLE] [/SNT]",
        "Output": "The Atatürk Monument is a bronze monument inaugurated on 27th July, 1932, in Izmir. It is found in Turkey, a country which has Ankara as its capital and Istanbul as its largest city. The leader of Turkey is called Ahmet Davutoğlu, and the currency is the Turkish lira."
    },
    2: {
        "Input": "[SNT] [TRIPLE] Turkey capital Ankara [/TRIPLE] [TRIPLE] Turkey largestCity Istanbul [/TRIPLE] [/SNT] [SNT] [TRIPLE] Turkey leader Ahmet_Davutoğlu [/TRIPLE] [TRIPLE] Turkey currency Turkish_lira [/TRIPLE] [/SNT] [SNT] [TRIPLE] Atatürk_Monument_(İzmir) location Turkey [/TRIPLE] [/SNT]",
        "Output": "The capital of Turkey is Ankara, although the largest city is Istanbul. The leader of Turkey is Ahmet Davutoglu and the currency is known as the Turkish lira. The Ataturk monument is located within the country."
    },
    3: {
        "Input": "[SNT] [TRIPLE] Antwerp_International_Airport cityServed Antwerp [/TRIPLE] [TRIPLE] Antwerp country Belgium [/TRIPLE] [TRIPLE] Belgium leaderName Philippe_of_Belgium [/TRIPLE] [TRIPLE] Belgium language French_language [/TRIPLE] [/SNT]",
        "Output": "Antwerp is served by Antwerp International Airport and is a popular tourism destination in Belgium where the leader is Philippe of Belgium and the French language is spoken."
    },
    4: {
        "Input": "[SNT] [TRIPLE] AWH_Engineering_College state Kerala [/TRIPLE] [TRIPLE] AWH_Engineering_College country India [/TRIPLE] [TRIPLE] AWH_Engineering_College established 2001 [/TRIPLE] [/SNT] [SNT] [TRIPLE] India river Ganges, India largestCity Mumbai [/TRIPLE] [/SNT] [SNT] Kerala leaderName Kochi [/TRIPLE] [/SNT]",
        "Output": "The AWH Engineering College in Kerala, India was established in 2001. The Ganges is a river in India and its largest city is Mumbai. The leader of Kerala is Kochi."
    },
    5: {
        "Input": "[SNT] [TRIPLE] Atlanta country United_States [/TRIPLE] [TRIPLE] United_States capital Washington_D.C. [/TRIPLE] [/SNT] [SNT] [TRIPLE] United_States ethnicGroup Asian_Americans [/TRIPLE] [/SNT]",
        "Output": "Atlanta is in the United States whose capital is Washington, D.C. Asian Americans are an ethnic group in the U.S."
    }
}

system_inst = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being socially unbiased and safe. \
If you're unsure about an answer, it's okay to skip it, and please ensure not to provide incorrect information. Additionally, responses should be concise and informative."
#instructions = f"Instruction: You will receive a list of triples containing subject, predicate and object. Write the triples as fluent and concise English text. Do not include any information that cannot be directly inferred from the given triples: "
instructions="I would like you to generate a fluent and concise summaries or text in English based on the triples provided. Below you may find examples of the input triples and the expected summary outputs. Do not omit any triple information in the text or include any information that cannot be directly inferred from the given triples."

instructions_struct="I would like you to generate a fluent and concise text in English based on the triples provided. Below you may find examples of the input triples and the expected textual outputs. Do not omit any triple information in the text or include any information that cannot be directly inferred from the given triples. Make sure to follow the sentence structure as found in the examples."


def generate_examples(data_examples, shot):
    examples = ''
    if shot == '0':
        examples = ''
    elif shot == '1':
        examples = f"\nInput: {data_examples[1]['Input']} \nOutput: {data_examples[1]['Output']} \n"
    elif shot == '3':
        for idx in range(1, 4):
            examples += f"\nInput: {data_examples[idx]['Input']} \nOutput: {data_examples[idx]['Output']} \n"
    elif shot == '5':
        for idx in range(1, 6):
            examples += f"\nInput: {data_examples[idx]['Input']} \nOutput: {data_examples[idx]['Output']} \n"
    else:
        raise ValueError("Number doesn't exist in the conditional statement. Please select these numbers: 0, 1, 3, and 5")

    return examples


def instruct_templates(model, shot, source, pipeline=False):
    global data_examples, data_examples_struct, system_inst, instructions, instructions_struct

    examples = generate_examples(data_examples_struct if pipeline else data_examples, shot)
    instructions = instructions_struct if pipeline else instructions
    if model in ['phi', 'cohere', 'gpt']:
        prompt = f'''{instructions}\nExamples:{examples}\nInput: {source}\nOutput:\n'''
    elif model in ['llama', 'mistral', 'vicuna']:
        prompt = f'''<s>[INST] <<SYS>> {system_inst}\n{instructions}\nExamples:{examples}<</SYS>>\nInput: {source}\nOutput:\n[/INST]'''
    elif model == 'gemma':
        prompt = f'''<start_of_turn> {system_inst}\n{instructions}\nExamples:{examples}\nInput: {source}\nOutput:\n<end_of_turn><start_of_turn>'''
    elif model == 'zephyr':
        prompt = f'''<|im_start|> {system_inst}\n<|im_end|><|im_start|>{instructions}\nExamples:{examples}\nInput: {source}\nOutput:\n<|im_end|><|im_start|>'''
    elif model == 'solar':
        prompt = f'''<s> {system_inst}\n{instructions}\nExamples:{examples}\nInput: {source}\nOutput:\n</s>'''
    elif model == 'olmo':
        prompt = f'''<|user|> {instructions}\nExamples:{examples}\nInput: {source}\nOutput: \n<|assistant|>'''
    else:
        raise ValueError("Model prompt doesn't exit in the conditional statement. Please select the right Model")

    return prompt if shot not in ['0', '1'] else prompt.replace('\nExamples:', '\nExample:') if shot == '1' else prompt.replace('\nExamples:', '')


# Function to write files
def write_file(write_path, result, mode='w'):
    with open(write_path, mode) as f:
        f.write('\n'.join(result))
