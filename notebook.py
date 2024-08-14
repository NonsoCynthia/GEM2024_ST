import os
import re
import time
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
#from transtruct import create_instruction, generate_examples

#specify a particular gpu to use if you have multiple ones. Here zero means your first GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#CUDA_VISIBLE_DEVICES=0


data_examples = {
    1: {
        'Input': 'Alfredo Zitarrosa died in Montevideo, Uruguay which is led by Raúl Fernando Sendic Rodríguez and Tabaré Vázquez.',
        'Arabic': ".توفي ألفريدو زيتاروزا في مونتيفيديو، أوروغواي التي يقودها راؤول فرناندو سينديتش رودريغيز وتاباري فاسكيز",
        'Korean': "영국의 수도는 런던으로, 돌로 만든 데드맨스 플랙(Dead Man's Plack)을 찾을 수 있습니다. Plack은 East Anglia의 Ealdorman인 Æthelwald에게 헌정되었습니다. 영국에서는 콘월어가 사용되며 영국 교회라는 종교가 확립되어 있습니다. 그 나라에서 발견되는 인종 그룹 중 하나는 영국계 아랍인입니다.",
        'Swahili': "Mji mkuu wa Uingereza ni London ambapo tunaweza kupata Plack ya Dead Man ambayo imetengenezwa kwa mawe. Plack imejitolea kwa Æthelwald, Ealdorman wa East Anglia. Lugha ya Cornish inazungumzwa nchini Uingereza na ina dini iliyoanzishwa inayoitwa Kanisa la Anglikana. Moja ya makabila yanayopatikana katika nchi hiyo ni Waarabu wa Uingereza."
    },
    2: {
        'Input': 'Angola International Airport is located at Ícolo e Bengo in Luanda province, Angola. The Airport is situated 159 meters above sea level and serves the city of Luanda.',
        'Arabic': ".يقع مطار أنغولا الدولي في ايكولو ايبينغو في مقاطعة لواندا، أنغولا. يقع المطار على ارتفاع 159 مترًا فوق مستوى سطح البحر ويخدم مدينة لواندا",
        'Korean': "앙골라 국제공항은 앙골라 루안다 지방의 이콜로 에 벤고에 위치해 있습니다. 공항은 해발 159미터에 위치해 있으며 루안다 시에 서비스를 제공합니다.",
        'Swahili': "Uwanja wa ndege wa Kimataifa wa Angola uko Ícolo e Bengo katika jimbo la Luanda, Angola. Uwanja wa ndege upo mita 159 juu ya usawa wa bahari na unahudumia jiji la Luanda."
    },
    3: {
        'Input': 'Akeem Adams, who plays for the Trinidad and Tobago national under-20 football team previously played for United Petrotrin FC whose ground is at Palo Seco.',
        'Arabic': ".أكيم آدامز، الذي يلعب لصالح منتخب ترينيداد وتوباغو لكرة القدم تحت 20 سنة، سبق له اللعب مع نادي يونايتد بيتروترين لكرة القدم الذي يقع ملعبه في بالو سيكو",
        'Korean': "트리니다드토바고 20세 이하 축구 국가대표팀에서 뛰고 있는 아킴 아담스는 팔로세코를 연고지로 하는 유나이티드 페트로트린 FC에서 선수 생활을 했습니다.",
        'Swahili': "Akeem Adams, anayechezea timu ya taifa ya vijana ya Trinidad na Tobago ya soka ya vijana chini ya umri wa miaka 20 hapo awali aliichezea United Petrotrin FC ambayo uwanja wake ni Palo Seco."
    },
    4: {
        'Input': 'The United States fighter pilot William Anders was born in British Hong Kong on the 17th of October, 1933. In 1963, he was chosen by NASA and became a crew member on Apollo 8.',
        'Arabic': ".8 ولد الطيار المقاتل الأمريكي ويليام أندرس في هونغ كونغ البريطانية في 17 أكتوبر 1933. وفي عام 1963، تم اختياره من قبل وكالة ناسا وأصبح أحد أفراد طاقم أبولو ",
        'Korean': "미국 전투기 조종사 윌리엄 앤더스는 1933년 10월 17일 영국령 홍콩에서 태어났어요. 1963년 NASA에 발탁되어 아폴로 8호의 승무원이 되었습니다.",
        'Swahili': "Rubani wa kivita wa Marekani William Anders alizaliwa Uingereza Hong Kong tarehe 17 Oktoba, 1933. Mnamo 1963, alichaguliwa na NASA na kuwa mwanachama wa wafanyakazi kwenye Apollo 8."
    },
    5: {
        'Input': "The capital of England is London where we can find the Dead Man's Plack which is made of stone. The Plack is dedicated to Æthelwald, Ealdorman of East Anglia. Cornish language is spoken in England and it has an established religion called the Church of England. One of the ethnic groups found in that country is the British Arabs.",
        'Arabic': ".عاصمة إنجلترا هي لندن حيث يمكننا العثور على نصب ديدمان بلاك تذكاري المصنوع من الحجر. المقام مخصص  للملك إيثلووف، زعيم و قائد من شرق أنجليا. يتم التحدث باللغة الكورنية في إنجلترا ولها دين راسخ يسمى كنيسة إنجلترا. إحدى المجموعات العرقية الموجودة في ذلك البلد هي العرب البريطانيين",
        'Korean': "영국의 수도 런던에는 돌로 만든 데드맨의 플랙이 있습니다. 이 플랙은 이스트 앵글리아의 에델발드에게 헌정되어 있어요. 영국에서는 콘월어를 사용하며 영국 국교회라는 종교가 확립되어 있습니다. 이 나라에서 발견되는 인종 그룹 중 하나는 영국 아랍인입니다.",
        'Swahili': "Mji mkuu wa Uingereza ni London ambapo tunaweza kupata Plack ya Dead Man ambayo imetengenezwa kwa mawe. Plack imejitolea kwa Æthelwald, Ealdorman wa East Anglia. Lugha ya Cornish inazungumzwa nchini Uingereza na ina dini iliyoanzishwa inayoitwa Kanisa la Anglikana. Moja ya makabila yanayopatikana katika nchi hiyo ni Waarabu wa Uingereza."
    }
}

def generate_examples(data_examples, output):
    examples = ''
    for idx in range(1, 6):
        examples += f"\nEnglish Text: {data_examples[idx]['Input']} \n\n{output} Text: {data_examples[idx][output]} \n\n"
    return examples


def create_instruction(lang, source):
    global data_examples
    if lang == 'ko':
        tgt_lang = "Korean"
    elif lang == 'sw':
        tgt_lang = "Swahili"
    elif lang == 'ar':
        tgt_lang = "Arabic"
    else:
        raise ValueError("Please use a valid language unicode")

    instruction = f"Translate the following English language text to {tgt_lang} language text. Provide only the translation. Follow the example below. \n\n######\n"
    examples = generate_examples(data_examples, tgt_lang)
    prompt = f'''{instruction} \nExamples:\n{examples}\nEnglish Text: {source}\n{tgt_lang} Text:\n'''
    return prompt


def read_file_lines(path: str) -> list:
    """
    Read lines from a file.

    Args:
    path (str): The path to the file.

    Returns:
    list: A list containing the lines of the file.
    """
    with open(path, 'r', encoding='utf-8') as file:
        contents = file.read()
        lines = [line.strip() for line in contents.split('\n')]
        if lines and lines[-1] == '':
            return lines[:-1]
        return lines


def write_files(write_path: str, result: list, mode: str = 'w') -> None:
    """
    Write contents to a file.

    Args:
    write_path (str): The path to write the file.
    result (list): The content to write to the file.
    mode (str, optional): The writing mode. Defaults to 'w'.
    """
    with open(write_path, mode) as f:
        f.write('\n'.join(result))
        
def write_file(write_path: str, result: str, mode: str = 'a') -> None:
    """
    Write contents to a file.

    Args:
    write_path (str): The path to write the file.
    results (list): The list of samples to write to the file.
    mode (str, optional): The writing mode. Defaults to 'a' (append).
    """
    with open(write_path, mode) as f:
        f.write(result + '\n')

def create_directory(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.

    Args:
    directory_path (str): The path of the directory to be created.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)      
        
def create_file(file_path: str) -> None:
    """
    Create a file if it doesn't exist.

    Args:
    file_path (str): The path of the file to be created.
    """
    with open(file_path, 'w') as f:
        pass  # This line does nothing, it's just to create an empty file

import shutil

def delete_directory(directory_path: str) -> None:
    """
    Delete a directory and its contents.

    Args:
    directory_path (str): The path of the directory to be deleted.
    """
    shutil.rmtree(directory_path)
    
    
import csv

def create_csv(csv_file):
    """
    Create a CSV file with two columns: 'Source' and 'Translation' and write the header row.

    Args:
    csv_file (str): The path of the CSV file to create.
    """
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Source', 'Translation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        
def append_to_csv(source, translation, csv_file):
    """
    Append a single sample to a CSV file with two columns: 'Source' and 'Translation'.

    Args:
    source (str): The source text.
    translation (str): The translation text.
    csv_file (str): The path of the CSV file to append.
    """
    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([source, translation])


class TranslationDataset(Dataset):
    def __init__(self, files_path):
        self.files_path = files_path
        self.files = [file for file in os.listdir(files_path) if file.endswith('.txt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        texts = read_file_lines(os.path.join(self.files_path, file_name))
        return texts, file_name


def generate(system, prompt):
    # Format message with the command-r-plus chat template
    messages = [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": prompt
            }
    ]
    text = pipe(messages, max_new_tokens=512)[0]['generated_text'][-1]['content']
    return text


def generate_(system, prompt):
    # Format message with the command-r-plus chat template
    messages = [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": prompt
            }
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    )
    gen_text = tokenizer.decode(gen_tokens[0])
    text = gen_text.split('<|CHATBOT_TOKEN|>')[1].replace('<|END_OF_TURN_TOKEN|>', '')
    return text


if __name__ == '__main__':
    
    model_id = "CohereForAI/c4ai-command-r-plus-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    pipe = pipeline("text-generation", model, tokenizer = tokenizer)

    datapath = '../results/cleaned/0_shot'
    dataset = TranslationDataset(datapath)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    #system prompt
    system = """You are an expert language translator able to communicate and translate one langage to another.\nYou should only translate the given text and provide only the translation."""

    #0_shot English to lang runs
    arabic_5_shot_path = f'{datapath}/swahili'
    create_directory(arabic_5_shot_path)

    for texts, file_name in tqdm(dataloader, desc="Processing files"):

        #path to create new file
        outfile = os.path.join(arabic_5_shot_path, file_name[0])
        csv_file = outfile.replace('.txt', '.csv')

        #create the file
        create_file(outfile)
        create_csv(csv_file)

        for text in tqdm(texts, desc="Translating texts"):
            text = text[0]
            #lang = 'ar' # 'sw', 'ko', 
            prompt = create_instruction('sw', text)
            result = generate(system, prompt)

            check1 = 'swahili:' #'korean:' #'arabic:'
            check2 = 'Swahili:' #'Korean:' #'Arabic:'
            check3 = 'swahili Text:' #'korean Text:' #'arabic Text:'
            check4 = 'Swahili text:' #'Korean text:' #'Arabic text:'

            if result.startswith(check1) or result.startswith(check2) or result.startswith(check4) or result.startswith(check3):
                result = result.replace(check1, '').replace(check2, '').replace(check3, '').replace(check4, '').strip()
            write_file(outfile, result, mode = 'a')
            append_to_csv(text, result, csv_file)
