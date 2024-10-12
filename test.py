import os
import re

#train_folder = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs/iSeVCs_for_train_programs'
train_folder = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs/iSeVCs_for_target_programs'

TOKEN_PATTERN = r'[@%]?\w+\*+|\w+|[\[\]{}(),=*]|[<>]|[0-9]+|#\d+'

def tokenize_line(llvm_code):
    line_tokens = re.findall(TOKEN_PATTERN, llvm_code)
    return line_tokens

record = 0
num_of_isevcs = 0
tot_tokens = 0
for i in range(len(os.listdir(train_folder))):
    filename_train = os.listdir(train_folder)[i]
    if filename_train.endswith(".txt"):
        filepath_train = os.path.join(train_folder, filename_train)
        with open(filepath_train, 'r') as file:
            iSeVC = []
            lines = file.readlines()
            reading_llvm_code = False
            for i in range(len(lines)):
                if i == len(lines) - 1:
                    break
                line = lines[i]
                next_line = lines[i + 1]
                if line.strip() == "" and 'define' in next_line:
                    reading_llvm_code = True
                    continue
                if line.strip() == "" and next_line.strip() == "":
                    labels_line = lines[i + 2]
                    reading_llvm_code = False
                    tokens_iSeVC = 0
                    for line in iSeVC:
                        tokens_iSeVC += len(line)
                    tot_tokens += tokens_iSeVC
                    if tokens_iSeVC > record:
                        record = tokens_iSeVC
                        print(record)
                    iSeVC = []
                    num_of_isevcs += 1
                    continue
                if reading_llvm_code:
                    tokenized_line = tokenize_line(line)
                    iSeVC.append(tokenized_line)

print('avarege tokens -> ' + str(tot_tokens/num_of_isevcs))
print('num of isevcs --> ' + str(num_of_isevcs))
print('longest isevc -> ' + str(record))
#296