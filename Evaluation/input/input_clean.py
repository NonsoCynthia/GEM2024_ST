import json

def read_file(path):
    if path.endswith(".json"):
        try:
            with open(path, "r", encoding='utf-8') as json_file:
                # Read the file line by line and parse each line as JSON
                data = []
                for line in json_file:
                    if line.strip():  # Only process non-empty lines
                        data.append(json.loads(line))
                # Assuming you need to work with the first object or merge them
                if len(data) == 1:
                    return data[0]
                else:
                    # If you need to merge or handle multiple objects, you can process `data` accordingly
                    raise ValueError("Multiple JSON objects found")
        except Exception as e:
            print(f"Error loading JSON: {e}")
            raise e
    else:
        with open(path, 'r', encoding='utf-8') as file:
            contents = file.read()
            lines = [line.strip() for line in contents.split('\n')]
            if lines and lines[-1] == '':
                return lines[:-1]
            return lines

input_files = ["counterfactual_struct.txt", "factual_struct.txt", "fictional_struct.txt", "counterfactual.json", "factual.json", "fictional.json"]

for struct in input_files:
    # Read the contents of the file
    doc = read_file(struct)
    
    if struct.endswith(".json"):
        # Assuming "triples" is the correct key for the JSON files
        doc = doc.get("triples", [])
        saved = struct.replace(".json", "_input.txt")
        
        # Replace patterns in the JSON text
        doc = [i.replace(" [/TRIPLE] [TRIPLE] ", ",") for i in doc]
        doc = [i.replace(" [/TRIPLE]", "") for i in doc]
        doc = [i.replace("[TRIPLE]", "") for i in doc]
    
    else:
        saved = struct.replace(".txt", "_input.txt")
        
        # Replace patterns in the TXT text
        doc = [i.replace(" [/TRIPLE] [TRIPLE] ", ",") for i in doc]
        doc = [i.replace(" [/TRIPLE]", "") for i in doc]
        doc = [i.replace(" [TRIPLE]", "") for i in doc]
    
    # Join the lines with '\n' and write to a new file
    with open(saved, 'w') as f:
        f.write('\n'.join(doc))

