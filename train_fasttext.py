import os
import re
import fasttext


main_folder = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs_training_fasttext/dataset'


#TOKEN_PATTERN = r'[@%]?\w+|[\[\]{}(),=*]|[<>]|[0-9]+|#\d+'
TOKEN_PATTERN = r'[@%]?\w+\*+|\w+|[\[\]{}(),=*]|[<>]|[0-9]+|#\d+'



def tokenize_line(line):
    return ' '.join(re.findall(TOKEN_PATTERN, line))



def process_and_tokenize_files(main_folder, output_file):

    with open(output_file, 'w') as outfile:

        for filename in os.listdir(main_folder):

            if filename.endswith(".txt"):
                filepath = os.path.join(main_folder, filename)
                print(f"Processing file: {filepath}")

                with open(filepath, 'r') as file:
                    lines = file.readlines()
                    reading_llvm_code = False
                    for i in range(len(lines)):
                        if i == len(lines) - 1:
                            break
                        line = lines[i]
                        next_line = lines[i + 1]
                        if line.strip() == "" and 'define' in next_line:
                            reading_llvm_code = True
                            #print('entro nel llvm')
                            continue
                        if line.strip() == "" and next_line.strip() == "":
                            reading_llvm_code = False
                            i += 1
                            #print('esco dal llvm')
                            continue
                        if reading_llvm_code:
                            tokenized_line = tokenize_line(line)
                            #print(tokenized_line + '\n')
                            outfile.write(tokenized_line + '\n')


#exit()


tokenized_corpus_file = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs_training_fasttext/tokenized_llvm_corpus.txt'


process_and_tokenize_files(main_folder, tokenized_corpus_file)


model = fasttext.train_unsupervised(tokenized_corpus_file, model='skipgram', dim=128)
#model = fasttext.train_unsupervised(tokenized_corpus_file, model='skipgram', dim=128, lr=0.025, epoch=10)


model.save_model('/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs_training_fasttext/fasttext_llvm_model.bin')


vector_alloca = model.get_word_vector('alloca')
print("Vector for 'alloca':", vector_alloca)
