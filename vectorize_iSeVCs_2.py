import numpy as np
import re
import fasttext
import gc
import os

sequence_length = 2618  #295 average / #2618 longest

embedding_dim = 64

TOKEN_PATTERN = r'[@%]?\w+\*+|\w+|[\[\]{}(),=*]|[<>]|[0-9]+|#\d+'

fasttext = fasttext.load_model(
    '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs_training_fasttext/fasttext_llvm_model.bin')


def tokenize_line(llvm_code):
    line_tokens = re.findall(TOKEN_PATTERN, llvm_code)
    return line_tokens


def extract_labels(line):
    line_strip = line.strip().strip('[]')
    string_vector = line_strip.split(',')
    vector = [int(num) for num in string_vector]
    return vector


def get_vulnerability_location_matrix(iSeVC, labels):
    vuln_loc_matrix = []
    if labels[0] == 0:
        tokens_num_in_iSeVC = 0
        for iSeVC_line in iSeVC:
            tokens_num_in_iSeVC += len(iSeVC_line)
        vuln_loc_matrix = [1] * tokens_num_in_iSeVC
    else:
        for j in range(len(iSeVC)):
            iSeVC_line = iSeVC[j]
            tokens_num_in_line = len(iSeVC_line)
            if j in labels:
                tokens_in_line = [1] * tokens_num_in_line
                vuln_loc_matrix.extend(tokens_in_line)
            else:
                tokens_in_line = [0] * tokens_num_in_line
                vuln_loc_matrix.extend(tokens_in_line)
    if len(vuln_loc_matrix) < sequence_length:
        padding_len = sequence_length - len(vuln_loc_matrix)
        padding = [0] * padding_len
        vuln_loc_matrix.extend(padding)
    else:
        vuln_loc_matrix = vuln_loc_matrix[:sequence_length]
    #print(np.array(vuln_loc_matrix).shape)
    return np.array(vuln_loc_matrix)


def vectorize_iSeVC(iSeVC):
    vectorized_iSeVC = []
    for line in iSeVC:
        for token in line:
            vectorized_token = fasttext.get_word_vector(token).tolist()
            vectorized_iSeVC.append(vectorized_token)
    if len(vectorized_iSeVC) < sequence_length:
        padding_len = sequence_length - len(vectorized_iSeVC)
        padding = [[0] * embedding_dim] * padding_len
        vectorized_iSeVC.extend(padding)
    else:
        vectorized_iSeVC = vectorized_iSeVC[:sequence_length]
    #print(np.array(vectorized_iSeVC).shape)
    return np.array(vectorized_iSeVC)


filepath = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs/iSeVCs_for_train_programs/PD_slices.txt'

#filepath = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs/iSeVCs_for_target_programs/PD_slices.txt'

outputpath = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs/Vectorized/Training/PD_slices/'

#outputpath = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs/Vectorized/Testing/PD_slices/'

with open(filepath, 'r') as file:
    print('Running')
    iSeVC = []
    line = file.readline()
    reading_llvm_code = False
    iSeVC_counter = 0
    while line:
        pos = file.tell()
        next_line = file.readline()
        file.seek(pos)
        if line.strip() == "" and 'define' in next_line:
            reading_llvm_code = True
            line = file.readline()
            continue
        if line.strip() == "" and next_line.strip() == "":
            if os.path.isfile(outputpath+f'{iSeVC_counter}.npz'):
                print('file '+f'{iSeVC_counter}.npz'+' exists\n\n\n\n\n')
                line = file.readline()
                iSeVC_counter += 1
                reading_llvm_code = False
                continue
            file.readline()
            labels_line = file.readline()
            if labels_line.strip() == "":
                break
            extracted_labels = extract_labels(labels_line.strip())
            #print(extracted_labels)
            vulnLocMatrix = get_vulnerability_location_matrix(iSeVC, extracted_labels)
            vulnLocMatrix = np.expand_dims(vulnLocMatrix, axis=0)
            vectorized_iSeVC = vectorize_iSeVC(iSeVC)
            iSeVC = []
            gc.collect()
            vectorized_iSeVC = np.expand_dims(vectorized_iSeVC, axis=0)
            label = np.array([])
            if extracted_labels[0] == 0:
                # label = np.array([0] * sequence_length)
                label = np.array([0])
            else:
                # label = np.array([1] * sequence_length)
                label = np.array([1])
            np.savez(outputpath + f'{iSeVC_counter}.npz',
                     iSeVC=vectorized_iSeVC,
                     vulnLocMatrix=vulnLocMatrix,
                     label=label)
            reading_llvm_code = False
            #print(iSeVC)
            del vulnLocMatrix
            del vectorized_iSeVC
            del label
            print('added ' + f'{iSeVC_counter}'+'.npz')
            iSeVC_counter += 1
            line = file.readline()
            continue
        if reading_llvm_code:
            if os.path.isfile(outputpath+f'{iSeVC_counter}.npz'):
                line = file.readline()
                continue
            tokenized_line = tokenize_line(line.strip())
            #print(tokenized_line)
            iSeVC.append(tokenized_line)
            line = file.readline()
            continue
        line = file.readline()

print('ho finito')
