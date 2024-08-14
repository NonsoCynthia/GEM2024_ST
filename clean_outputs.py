import os

def write_file(write_path, result, mode='w'):
    with open(write_path, mode) as f:
        f.write('\n'.join(result))

def read_file_lines(path):
    with open(path, 'r', encoding='utf-8') as file:
        contents = file.read()
        lines = [line.strip() for line in contents.split('\n')]
        if lines and lines[-1] == '':
            return lines[:-1]
        return lines

# Define the input and output folders
input_folders = ['0_shot', '5_shot']
output_folder = 'cleaned_later'

for folder in input_folders:
    input_folder_path = os.path.join("results", folder)
    output_folder_path = os.path.join("results", output_folder, folder)
    
    # Create the output folder if it does not exist
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Check if the input folder exists
    if os.path.exists(input_folder_path):
        files = os.listdir(input_folder_path)
        
        # Filter files to consider only text files
        files = [file for file in files if file.endswith('.txt')]

        # Process each file
        for file_name in files:
            file_path = os.path.join(input_folder_path, file_name)

            #print(file_path)

            # Read lines from the file
            lines = read_file_lines(file_path)
            
            cleaned_lines = []
            if "mistral" in file_name:
                # Extract relevant information from the file
                lines = [line.split('Output:  [/INST] ')[-1].strip() if 'Output:  [/INST] ' in line else line for line in lines]
                lines = [line.split(' Input: ')[0].strip() if ' Input: ' in line else line for line in lines]
                lines = [line.split('Output: ')[-1].strip() if 'Output: ' in line else line for line in lines]
                #lines = [line.split(' Input: ')[0].strip() if ' Input: ' in line else line for line in lines]
                lines = [line.split(' [SNT] ')[0].strip() if ' [SNT] ' in line else line for line in lines]
                lines = [line.split('[Note ')[0].strip() if '[Note ' in line else line for line in lines]
                cleaned_lines.extend(lines)
                
            elif 'phi' in file_name:
                for line in lines:
                    if '   Output:' in line:
                        line = line.split('Output: ')[-1].strip()
                    elif '   Input:' in line:
                        line = line.split('   Input:')[-1].strip()
                    elif '#' in line:
                        line = line.split('#')[0].strip()
                    else:
                        line = line.strip()
                    cleaned_lines.append(line)  # Indentation fixed here
            elif 'gpt' in file_name:
                for line in lines:
                    line = line.strip()
                    cleaned_lines.append(line)  # Indentation fixed here

            # Extend the result list instead of appending
            output_file_path = os.path.join(output_folder_path, file_name)
            #print(output_file_path)
            write_file(output_file_path, cleaned_lines, mode='w')


# Print the last 10 lines of the result for verification
#for folder in input_folders:
    #output_folder_path = os.path.join("results", output_folder, folder)
    #files = os.listdir(output_folder_path)
    #for file_name in files:
        #output_file_path = os.path.join(output_folder_path, file_name)
        #print(f"Contents of {output_file_path}:")
        #with open(output_file_path, 'r') as f:
            #print(f.read())
