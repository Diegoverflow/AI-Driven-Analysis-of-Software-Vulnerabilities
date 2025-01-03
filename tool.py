from keras.api.models import load_model
from keras.api.models import Model
from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np
import re
import fasttext
from keras import config

config.enable_unsafe_deserialization()

k_value = 1

sequence_length = 2618

embedding_dim = 32

model = load_model('/home/httpiego/PycharmProjects/VulDeeDiegator/trained_models/gru/trained_model_2.keras')#, custom_objects={'k_max_pooling': k_max_pooling(x=embedding_dim)})

iSeVC_path = '/home/httpiego/PycharmProjects/VulDeeDiegator/TestPrograms/bof/strcpy/iSeVC/bof_strcpy_iSeVC'

sequence_length = 2618  #295 average / #2618 longest

embedding_dim = 32

TOKEN_PATTERN = r'[@%]?\w+\*+|\w+|[\[\]{}(),=*]|[<>]|[0-9]+|#\d+'

fasttext = fasttext.load_model(
    '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs_training_fasttext/fasttext_llvm_model_' + str(embedding_dim) + '.bin')

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

# Custom function for k-max pooling to ensure proper deserialization
def k_max_pooling(x):
    return tf.math.top_k(x, k=k_value).values

with open(iSeVC_path, 'r') as iSeVC:
    vectorized_iSeVC = vectorize_iSeVC(iSeVC.read())
    vulnLocMatrix = [1] * 2618

    #iSeVC = np.expand_dims(vectorized_iSeVC, axis=0)  # Add batch dimension
    #vulnLocMatrix = np.expand_dims(vulnLocMatrix, axis=0)  # Add batch dimension

    # Start with the input layer
    inputs = model.input

    layer_to_replace = 'k_max_pooling'
    # Traverse the layers and reconstruct the model
    x = inputs
    for layer in model.layers:

        if layer.name == layer_to_replace:
            # Replace the layer with a new one
            x = layers.Lambda(lambda x: tf.math.top_k(x, k=k_value).values,
                                 output_shape=(k_value, embedding_dim),
                                 name='k_max_pooling')(x)
        else:
            # Add the existing layer
            x = layer(x)

    new_model = Model(inputs=inputs, outputs=x)

    for layer in new_model.layers:
            layer.set_weights(model.get_layer(layer.name).get_weights())

    # Create the new model

    predictions = model.predict([iSeVC, vulnLocMatrix])

    print(f"Prediction: {predictions}")

