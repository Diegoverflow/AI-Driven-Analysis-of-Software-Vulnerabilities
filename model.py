import tensorflow as tf
from tensorflow.keras import layers, models
import os
import re
import fasttext

# Hyperparameters
sequence_length = 50  # This is θ, the fixed length for the iSeVC (after padding or truncation)
embedding_dim = 128  # The dimensionality of the token embeddings (e.g., 128)
rnn_units = 64  # The number of units in the LSTM or GRU layer
k_value = 5  # The κ value for k-max pooling (hyperparameter)
num_classes = 1  # Output (binary classification - vulnerable or non-vulnerable)


# Function to build the BRNN-vdl neural network
def build_vuldee_model(sequence_length, embedding_dim, rnn_units, k_value, num_classes):
    # Input layer
    inputs = layers.Input(shape=(sequence_length, embedding_dim), name='input_layer')

    # Bidirectional RNN (LSTM or GRU)
    brnn = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True), name='bidirectional_rnn')(inputs)

    # Dense layer to reduce dimensionality
    dense = layers.Dense(rnn_units, activation='relu', name='dense_layer')(brnn)

    # Activation layer
    activations = layers.Activation('relu', name='activation_layer')(dense)

    # Vulnerability Location Matrix (assume we compute it elsewhere, for this example it's all ones)
    vulnerability_location_matrix = layers.Input(shape=(sequence_length,), name='vulnerability_location_input')

    # Use a Lambda layer instead of tf.expand_dims()
    expanded_vulnerability_location_matrix = layers.Lambda(lambda x: tf.expand_dims(x, -1))(
        vulnerability_location_matrix)

    # Multiply layer (to highlight vulnerable tokens)
    multiply = layers.Multiply(name='multiply_layer')([activations, expanded_vulnerability_location_matrix])

    # κ-max pooling layer
    k_max_values = layers.Lambda(lambda x: tf.math.top_k(x, k=k_value).values, name='k_max_pooling')(multiply)

    # Average pooling layer
    average_pooling = layers.GlobalAveragePooling1D(name='average_pooling_layer')(k_max_values)

    # Output layer: Binary classification (vulnerable or non-vulnerable)
    output = layers.Dense(num_classes, activation='sigmoid', name='output_layer')(average_pooling)

    # Define the model
    model = models.Model(inputs=[inputs, vulnerability_location_matrix], outputs=output, name='VulDeeLocator_NN')

    return model


# Build the model
vuldee_model = build_vuldee_model(sequence_length, embedding_dim, rnn_units, k_value, num_classes)

# Compile the model
vuldee_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
vuldee_model.summary()

fasttext = fasttext.load_model(
    '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs_training_fasttext/fasttext_llvm_model.bin')

TOKEN_PATTERN = r'[@%]?\w+\*+|\w+|[\[\]{}(),=*]|[<>]|[0-9]+|#\d+'


def tokenize_line(llvm_code):
    line_tokens = re.findall(TOKEN_PATTERN, line)
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
        for j in range(iSeVC):
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
        vuln_loc_matrix = vuln_loc_matrix.extend(padding)
    else:
        vuln_loc_matrix = vuln_loc_matrix[:sequence_length]
    return vuln_loc_matrix


def train_model(iSeVC, labels):
    vulnerability_location_matrix = get_vulnerability_location_matrix(iSeVC, labels)
    vectorized_iSeVC = []
    for line in iSeVC:
        for token in line:
            vectorized_token = fasttext.get_word_vector(token)
            vectorized_iSeVC.append(vectorized_token)
    if len(vectorized_iSeVC) < sequence_length:
        padding_len = sequence_length - len(vectorized_iSeVC)
        padding = [0] * padding_len
        vectorized_iSeVC = vectorized_iSeVC.extend(padding)
    else:
        vectorized_iSeVC = vectorized_iSeVC[:sequence_length]

    vuldee_model.fit([vectorized_iSeVC, vulnerability_location_matrix], labels, epochs=5, batch_size=32)


training_folder = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs/iSeVCs_for_target_programs'
fasttext = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs_training_fasttext/fasttext_llvm_model.bin'

for filename in os.listdir(training_folder):
    # Check if the file has a .txt extension (adjust as per your file types)
    if filename.endswith(".txt"):
        filepath = os.path.join(training_folder, filename)
        print(f"Processing file: {filepath}")

        with open(filepath, 'r') as file:
            iSeVC = []
            labels = []
            lines = file.readlines()
            reading_llvm_code = False
            for i in range(len(lines)):
                if i == 50:
                    break
                if i == len(lines) - 1:
                    break
                line = lines[i]
                next_line = lines[i + 1]
                if line.strip() == "" and 'define' in next_line:
                    reading_llvm_code = True
                    continue
                if line.strip() == "" and next_line.strip() == "":
                    labels_line = lines[i + 2]
                    labels = extract_labels(labels_line)
                    train_model(iSeVC, labels)
                    reading_llvm_code = False
                    iSeVC = []
                    labels = []
                    #i += 1
                    continue
                if reading_llvm_code:
                    tokenized_line = tokenize_line(line)
                    print(tokenized_line)
                    iSeVC.append(tokenized_line)

vuldee_model.save('/home/httpiego/PycharmProjects/VulDeeDiegator/trained_model.h5')
