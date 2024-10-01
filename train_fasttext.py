import os
import re
import fasttext

# Step 1: Define the path to your main folder
main_folder = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs_training_fasttext/dataset'

# Step 2: Define a regular expression pattern for tokenization (you can adjust this as needed)
#TOKEN_PATTERN = r'[@%]?\w+|[\[\]{}(),=*]|[<>]|[0-9]+|#\d+'
TOKEN_PATTERN = r'[@%]?\w+\*+|\w+|[\[\]{}(),=*]|[<>]|[0-9]+|#\d+'


# Step 3: Function to tokenize a line of LLVM IR code
def tokenize_line(line):
    return ' '.join(re.findall(TOKEN_PATTERN, line))


# Step 4: Function to process and tokenize all files in the main folder
def process_and_tokenize_files(main_folder, output_file):
    # Open a temporary file for writing the tokenized content
    with open(output_file, 'w') as outfile:
        # Loop through each file in the main folder
        for filename in os.listdir(main_folder):
            # Check if the file has a .txt extension (adjust as per your file types)
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

# Step 5: Define the path for the combined tokenized corpus file
tokenized_corpus_file = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs_training_fasttext/tokenized_llvm_corpus.txt'

# Step 6: Process all files and write tokenized content to the corpus file
process_and_tokenize_files(main_folder, tokenized_corpus_file)

# Step 7: Train FastText on the tokenized corpus
model = fasttext.train_unsupervised(tokenized_corpus_file, model='skipgram', dim=128)
#model = fasttext.train_unsupervised(tokenized_corpus_file, model='skipgram', dim=128, lr=0.025, epoch=10)

# Step 8: Save the FastText model
model.save_model('/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs_training_fasttext/fasttext_llvm_model.bin')

# Step 9: Example: Get vector for a specific token (e.g., 'alloca')
vector_alloca = model.get_word_vector('alloca')
print("Vector for 'alloca':", vector_alloca)
