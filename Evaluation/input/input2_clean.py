import os
import json
import pandas as pd

# Read data from files
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


ftask = ["factual", "fictional", "counterfactual"]
data = {}
    
for t in ftask:
    data[t] = pd.read_json(os.path.join(f"{t}.json"), encoding='utf-8-sig', lines=True, orient="records")
    struct_file = read_file(os.path.join(f"{t}_struct.txt"))
    data[f"{t}_struct"] = pd.DataFrame({"triples": struct_file})

    # Write DataFrame contents to text files
for t, df in data.items():
    df['triples'] = [i.replace(" [/TRIPLE] [TRIPLE] ", ", ") for i in df['triples']]
    df['triples'] = [i.replace(" [/TRIPLE]", "") for i in df['triples']]
    df['triples'] = [i.replace("[TRIPLE] ", "") for i in df['triples']]
    #df['triples'] = [ i.replace("  ", " ") for i in df['triples']]

    output_file_path = f"{t}_input.txt"
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(df['triples']))
