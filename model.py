import tensorflow as tf
from tensorflow.keras import layers, models
import os
import re
import fasttext
from sklearn.model_selection import train_test_split
import numpy as np

# Hyperparameters
sequence_length = 295 #--> average / #2618 longest
embedding_dim = 128
rnn_units = 64
k_value = 5
num_classes = 1


# Function to build the BRNN-vdl neural network
def build_vuldee_model(sequence_length, embedding_dim, rnn_units, k_value, num_classes):
    inputs = layers.Input(shape=(sequence_length, embedding_dim), name='input_layer')

    brnn = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True), name='bidirectional_rnn')(inputs)

    dense = layers.Dense(rnn_units, activation='relu', name='dense_layer')(brnn)

    activations = layers.Activation('relu', name='activation_layer')(dense)

    vulnerability_location_matrix = layers.Input(shape=(sequence_length,), name='vulnerability_location_input')

    expanded_vulnerability_location_matrix = layers.Lambda(lambda x: tf.expand_dims(x, -1))(
        vulnerability_location_matrix)

    multiply = layers.Multiply(name='multiply_layer')([activations, expanded_vulnerability_location_matrix])

    k_max_values = layers.Lambda(lambda x: tf.math.top_k(x, k=k_value).values, name='k_max_pooling')(multiply)

    average_pooling = layers.GlobalAveragePooling1D(name='average_pooling_layer')(k_max_values)

    output = layers.Dense(num_classes, activation='sigmoid', name='output_layer')(average_pooling)

    model = models.Model(inputs=[inputs, vulnerability_location_matrix], outputs=output, name='VulDeeLocator_NN')

    return model


vuldee_model = build_vuldee_model(sequence_length, embedding_dim, rnn_units, k_value, num_classes)

vuldee_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

vuldee_model.summary()

fasttext = fasttext.load_model(
    '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs_training_fasttext/fasttext_llvm_model.bin')

TOKEN_PATTERN = r'[@%]?\w+\*+|\w+|[\[\]{}(),=*]|[<>]|[0-9]+|#\d+'


def tokenize_line(llvm_code):
    line_tokens = re.findall(TOKEN_PATTERN, llvm_code)
    return line_tokens


def extract_labels(line):
    line_strip = line.strip().strip('[]')
    string_vector = line_strip.split(',')
    vector = [int(num) for num in string_vector]
    return np.array(vector)


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
    return np.array(vuln_loc_matrix)


def vectorize_iSeVC(iSeVC):
    vectorized_iSeVC = []
    for line in iSeVC:
        for token in line:
            vectorized_token = fasttext.get_word_vector(token).tolist()
            vectorized_iSeVC.append(vectorized_token)
    if len(vectorized_iSeVC) < sequence_length:
        padding_len = sequence_length - len(vectorized_iSeVC)
        padding = [[0]*embedding_dim] * padding_len
        vectorized_iSeVC.extend(padding)
    else:
        vectorized_iSeVC = vectorized_iSeVC[:sequence_length]
    return np.array(vectorized_iSeVC)


def get_iSeVCs_vulnLocMatrixes_labels(filepath):
    iSeVCs = []
    vulnLocMatrixes = []
    labels = []
    with open(filepath, 'r') as file:
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
                extracted_labels = extract_labels(labels_line)
                print(extracted_labels)
                #train_model(iSeVC, labels)
                vectorized_iSeVC = vectorize_iSeVC(iSeVC)
                iSeVCs.append(vectorized_iSeVC)
                vulnLocMatrixes.append(get_vulnerability_location_matrix(iSeVC, extracted_labels))
                if extracted_labels[0] == 0:
                    labels.append(0)
                else:
                    labels.append(1)
                reading_llvm_code = False
                iSeVC = []
                continue
            if reading_llvm_code:
                tokenized_line = tokenize_line(line)
                print(tokenized_line)
                iSeVC.append(tokenized_line)
    iSeVCs_vulnLocMatrixes_labels = {'iSeVCs': np.array(iSeVCs), 'vulnLocMatrixes': np.array(vulnLocMatrixes), 'labels': np.array(labels)}
    return iSeVCs_vulnLocMatrixes_labels


train_folder = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs/iSeVCs_for_train_programs'
test_folder = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs/iSeVCs_for_target_programs'

for i in range(len(os.listdir(train_folder))):
    filename_train = os.listdir(train_folder)[i]
    filename_test = os.listdir(test_folder)[i]
    if filename_train.endswith(".txt") and filename_test.endswith(".txt"):
        filepath_train = os.path.join(train_folder, filename_train)
        filepath_test = os.path.join(test_folder, filename_test)
        print(f"Processing train file: {filepath_train}")
        print(f"Processing test file: {filename_test}")
        train_iSeVCs_vulnLocMatrixes_labels = get_iSeVCs_vulnLocMatrixes_labels(filepath_train)
        iSeVCs = train_iSeVCs_vulnLocMatrixes_labels['iSeVCs']
        vulnerability_location_matrixes = train_iSeVCs_vulnLocMatrixes_labels['vulnLocMatrixes']
        labels = train_iSeVCs_vulnLocMatrixes_labels['labels']
        print('shape iSeVCs --> ' + str(iSeVCs.shape))
        print('shape vulnerability_location_matrixes --> ' + str(vulnerability_location_matrixes.shape))
        print('shape labels --> ' + str(labels.shape))
        iSeVCs_train, iSeVCs_val, vuln_loc_train, vuln_loc_val, labels_train, labels_val = train_test_split(
            iSeVCs, vulnerability_location_matrixes, labels, test_size=0.2, random_state=42
        )
        exit()
        vuldee_model.fit([iSeVCs_train, vuln_loc_train],
                         labels_train,
                         epochs=10,
                         batch_size=32,
                         validation_data=([iSeVCs_val, vuln_loc_val], labels_val)
                         )

        test_iSeVCs_vulnLocMatrixes_labels = get_iSeVCs_vulnLocMatrixes_labels(filepath_test)
        test_loss, test_accuracy = vuldee_model.evaluate([test_iSeVCs_vulnLocMatrixes_labels['iSeVCs'],
                                                          test_iSeVCs_vulnLocMatrixes_labels['vulnLocMatrixes']],
                                                         test_iSeVCs_vulnLocMatrixes_labels['labels'],
                                                         batch_size=32)

        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

vuldee_model.save('/home/httpiego/PycharmProjects/VulDeeDiegator/trained_model.h5')
