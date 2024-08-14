import os
import re
import json
import re


def ord_tags(text):
    s_to_remove = ['[TRIPLE]', '[/TRIPLE]', '/TRIPLE','TRIPLE', '[', ']', ',', '.', "'", '"']
    #s_to_remove = [',', '.']
    for punctuation in s_to_remove:
        text = text.replace(punctuation, '')  
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def struct_tags(contents):
    replacements = {
        '[ SNT)': '[SNT]', '[ /SNT)': '[/SNT]', '[SNT)': '[SNT]', '[/SNT)': '[/SNT]',
        '[ SNT] ': '[SNT]', '[/ SNT] ': '[/SNT]', '[ Snt] ': '[SNT]', '[/ Snt] ': '[/SNT]',
        '[sNT)': '[SNT]', '[/sNT)': '[/SNT]', '[T] ': '[SNT]', '[/T] ': '[/SNT]',
        '[SNT?]': '[SNT]', '[/SNT?]': '[/SNT]', '[SOD]': '[SNT]', '[/SOD]': '[/SNT]',
        ' SNT]': '[SNT]', ' /SNT]': '[/SNT]', '[SENT]': '[SNT]', '[/SENT]': '[/SNT]',
        '[S NT]': '[SNT]', '[/S NT]': '[/SNT]', '[SVP]': '[SNT]', '[/SVP]': '[/SNT]',
        '[SDP]': '[SNT]', '[/SDP]': '[/SNT]', '[SNT...]': '[SNT]', '[/SNT...]': '[/SNT]',
        '[SOTT]': '[SNT]', '[/SOTT]': '[/SNT]', '[sNT]': '[SNT]', '[/sNT]': '[/SNT]',
        '[SOT]': '[SNT]', '[/SOT]': '[/SNT]', '[SNS]': '[SNT]', '[/SNS]': '[/SNT]',
        '], [': '] [', ']. [': '] [', '],':']', '].':']', '[[[': '[', '[[': '[',
        '[SNT].': '[SNT]', '[/SNT].': '[SNT]', '(SNT)':'[SNT]', '(/SNT)':'[/SNT]',
        '[SNT],':'[SNT]', '[/SNT],':'[/SNT]', '[SNT].':'[SNT]', '[/SNT].':'[/SNT]',
        '[SNA]':'[SNT]', '[/SNA]': '[/SNT]', '[SNS]':'[SNT]', '[/SNS]':'[/SNT]',
        "[SNT']":'[SNT]', "[/SNT']":'[/SNT]', '[SS]':'[SNT]', '[/SS]':'[/SNT]',
        '[SUNT]':'[SNT]', '[/SUNT]':'[/SNT]', '[DNT|':'[SNT]', '[/DNT|':'[/SNT]',
        '[SENT]':'[SNT]', '[/SENT]':'[/SNT]','[SNOR]':'[SNT]', '[/SNOR]':'[/SNT]',

        }

    xcontents = contents
    for old, new in replacements.items():
        xcontents = xcontents.replace(old, new)

    pattern = re.compile(r'\[\s*(\/?\w+)\s*\]')
    pattern2 = re.compile(r'\[\s*(\/?\s*[\w\.,]+)\s*\]\s*[\.,]?\s*')

    cleaned_text = re.sub(pattern, lambda m: f"[{'/' if m.group(1).startswith('/') else ''}SNT] ", xcontents)
    cleaned_text = re.sub(pattern2, lambda m: f"[{'/' if m.group(1).startswith('/') else ''}SNT] ", cleaned_text)
    cleaned_text = re.sub(r'\[SNT\]([^\s])', r'[SNT] \1', cleaned_text)  # Add space after [SNT] if not followed by a space
    cleaned_text = cleaned_text.replace('  ', ' ')
    # Split the text into lines
    #cleaned_text_lines = cleaned_text.strip().split('\n')

    return cleaned_text

def read_file(path):
    if path.endswith(".json"):
        try:
            with open(path, "r", encoding='utf-8') as json_file:
                data = json.load(json_file)
        except:
            json_string = open(path).read()
            data = json.loads(json_string)
        return data
    else:
        with open(path, 'r', encoding='utf-8') as file:
            contents = file.read()
            contents = contents.replace('<', '[').replace('>', ']')
            lines = [line.strip() for line in contents.split('\n')]
            if lines and lines[-1] == '':
                return lines[:-1]
            return lines

# Read data from files
def read_file_map(path, task):
    if path.endswith(".json"):
        with open(path, "r", encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data
    else:
        with open(path, 'r', encoding='utf-8') as file:
            contents = file.read()
            if task == "structuring":
                lines = [struct_tags(line.strip()) for line in contents.split('\n')]
                #print(lines)
            else:
                #lines = [ord_tags(line.strip()) for line in contents.split('\n')]
                lines = [line.strip() for line in contents.split('\n')]
                
            if lines and lines[-1] == '':
                return lines[:-1]
            return lines

def write_file(write_path, result):
    with open(write_path, 'w') as f:
        f.write('\n'.join(result))

def split_triples(text):
    triples, triple = [], []
    for w in text:
        if w not in ['[TRIPLE]', '[/TRIPLE]']:
            triple.append(w)
        elif w == '[/TRIPLE]':
            triples.append(triple)
            triple = []
    return triples

def join_triples(triples):
    result = []
    for triple in triples:
        result.append('[TRIPLE]')
        result.extend(triple)
        result.append('[/TRIPLE]')
    return result

def join_struct(sentences):
    result = []
    for sentence in sentences:
        result.append('[SNT]')
        for triple in sentence:
            result.append('[TRIPLE]')
            result.extend(triple)
            result.append('[/TRIPLE]')
        result.append('[/SNT]')
    return result

def orderout2structin(ordering_out, triples):
    ord_triples = []
    if len(triples) == 1:
        ord_triples.extend(triples)
    else:
        added = []
        for predicate in ordering_out:
            for i, triple in enumerate(triples):
                if predicate.strip() == triple[1].strip() and i not in added:
                    ord_triples.append(triple)
                    added.append(i)
                    break
    return ' '.join(join_triples(ord_triples))
                    

# EDIT THIS CODE FROM HERE!!!!
def structout2lexin(struct_out, triples):
    sentences, snt = [], []
    for w in struct_out:
        if w.strip() not in ['[SNT]', '[/SNT]']:
            snt.append(w.strip())

        if w.strip() == '[/SNT]':
            sentences.append(snt)
            snt = []

    struct, struct_unit = [], []
    if len(triples) == 1:
        struct.append(triples)
    else:
        added = []
        for snt in sentences:
            for predicate in snt:
                for i, triple in enumerate(triples):
                    if predicate.strip() == triple[1].strip() and i not in added:
                        struct_unit.append(triple)
                        added.append(i)
                        break
            struct.append(struct_unit)
            struct_unit = []
    return ' '.join(join_struct(struct))

# STOP HERE!!!!

def run(out_path, entries_path, pre_task):
    outputs = [out.split() for out in out_path]
    entries = [split_triples(t.split()) for t in entries_path]
    #print(outputs, '##',entries)
    
    if len(outputs) != len(entries):
        print(f"Length of {pre_task} outputs: {len(outputs)}")
        print(f"Length of entries: {len(entries)}")
        #print(outputs, entries)
        raise ValueError("Number of outputs does not match number of entries")

    add = [] 
    for i in range(len(outputs)):
        if pre_task == "ordering":
            add.append(orderout2structin(ordering_out=outputs[i], triples=entries[i]))
        elif pre_task == "structuring":
            add.append(structout2lexin(struct_out=outputs[i], triples=entries[i]))
        else:
            raise ValueError("Invalid pre_task value")
    print(add[384], '\n', add[703], '\n', add[1132], '\n', add[1407])
    
    for i in range(len(outputs)):
        if add[i] == "":
            print(i, outputs[i], entries[i])
            if pre_task == "ordering":
                add[i] = entries_path[i]
            else:
                add[i] = "[SNT] " + entries_path[i] + " [/SNT]"

    return add

# Read the JSON file
#pre_task="ordering"
pre_task="structuring"
dir_ =f"/home/cosuji/spinning-storage/cosuji/NLG_Exp/gem/results"
for files in ['factual', 'fictional', 'counterfactual', 'simon']:
    if files == 'simon':
        data = read_file_map(os.path.join(dir_, f'{pre_task}/{files}_{pre_task}.json'), pre_task)
        data2 = read_file_map(os.path.join(dir_, f'{pre_task}/{files}_{pre_task}.txt'), pre_task)
        write_path_test = os.path.join(dir_, f'{pre_task}/{files}_{pre_task}.mapped')
        #strg = f'/home/cosuji/spinning-storage/cosuji/NLG_Exp/gem/results/ordering/{files}_ordering.mapped'
        # Assuming each item in the list is a dictionary with "input" and "pred" keys
        input_data = [item["input"] for item in data]
        pred_data = [item["pred"] for item in data]

        print(len(input_data), len(pred_data))
        
        #if pre_task == 'structuring':
        #out_path_test = [struct_tags(text.strip()) for text in pred_data]
        #else:
        out_path_test = [re.sub(r'\s+', ' ', text.strip()) for text in pred_data]
        print(out_path_test[384])
        entries_path_test = [re.sub(r'\s+', ' ', text).replace('[/TRIPLE],', '[/TRIPLE]') for text in input_data]
        print(entries_path_test[384])
        result_test = run(out_path=out_path_test, entries_path=entries_path_test, pre_task=pre_task)
        # Write the result to a file
        write_file(write_path_test, result_test)

        #for i in range(len(pred_data)):
        #for i in [287, 453, 1161, 1749]:
        #for i in [184, 253, 838]:
            #print(pred_data[i])
            #print(f"{i-1}, Mapped:{result_test[i-1]}; Result:{data2[i-1]}; Input:{input_data[i-1]}")
            #print(f"{i}, Mapped:{result_test[i]},Result:{data2[i]}, Input:{input_data[i]}")
