import gc

import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
sequence_length = 2618  #295 average / #2618 longest
embedding_dim = 64
rnn_units = 64
k_value = 5
num_classes = 1
batch_size = 128


# Function to build the BRNN-vdl neural network
def build_vuldee_model(sequence_length, embedding_dim, rnn_units, k_value, num_classes):

    inputs = layers.Input(shape=(sequence_length, embedding_dim), name='input_layer')

    brnn = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True), name='bidirectional_rnn')(inputs)

    dense = layers.Dense(rnn_units, activation='relu', name='dense_layer')(brnn)

    #dropout = layers.Dropout(0.5)(dense)

    #activations = layers.Activation('relu', name='activation_layer')(dropout)

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


train_folder = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs/Vectorized/Training/'
test_folder = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs/Vectorized/Testing/'

train_files = os.listdir(train_folder + 'AD_slices/')

print(len(train_files))


def load_data(file_name):
    # Load the data from the .npz file
    data = np.load(file_name)#, allow_pickle=True)

    iSeVC = data['iSeVC']
    vulnLocMatrix = data['vulnLocMatrix']
    label = data['label']

    return iSeVC, vulnLocMatrix, label

def create_batch(file_list):
    iSeVCs = []
    vulnLocMatrixes = []
    labels = []

    for file_name in file_list:
        iSeVC, vulnLocMatrix, label = load_data(file_name)
        #iSeVC = np.squeeze(iSeVC, axis=1)
        iSeVCs.append(iSeVC)
        #vulnLocMatrix = np.squeeze(vulnLocMatrix, axis=1)
        vulnLocMatrixes.append(vulnLocMatrix)
        labels.append(label)

    # Convert lists into NumPy arrays
    iSeVCs = np.array(iSeVCs)  # Shape: (64, seq_len, input_dim)
    iSeVCs = np.squeeze(iSeVCs, axis=1)
    vulnLocMatrixes = np.array(vulnLocMatrixes)    # Shape: (64, input_vec_dim)
    vulnLocMatrixes = np.squeeze(vulnLocMatrixes, axis=1)
    labels = np.array(labels)                  # Shape: (64,)

    return iSeVCs, vulnLocMatrixes, labels


all_losses = []
all_accuracies = []
for i in range(len(os.listdir(train_folder))):

    #TRAIN
    train_subfolder = os.listdir(train_folder)[i]
    train_files_lenght = len(os.listdir(train_folder + train_subfolder))
    start_index = 0
    last_index = batch_size
    while True:
        gc.collect()
        files_in_batch = []
        #print(start_index)
        print('Path: ' + train_folder + train_subfolder)
        print(f'batch: {start_index} - {last_index}')
        for j in range(start_index, last_index):
            files_in_batch.append(train_folder + train_subfolder + f'/{j}.npz')
        iSeVCs, vulnLocMatrixes, labels = create_batch(files_in_batch)
        #print(iSeVCs.shape)
        #print(vulnLocMatrixes.shape)
        #print(labels.shape)
        history = vuldee_model.fit([iSeVCs, vulnLocMatrixes], labels, epochs=10, batch_size=batch_size)
        all_losses.extend(history.history['loss'])
        all_accuracies.extend(history.history['accuracy'])
        del files_in_batch, iSeVCs, vulnLocMatrixes, labels
        start_index = last_index
        if start_index == train_files_lenght-1:
            break
        last_index += batch_size
        if last_index > (train_files_lenght-1):
            last_index = train_files_lenght-1

    #EVALUATE
    test_subfolder = os.listdir(test_folder)[i]
    test_files_lenght = len(os.listdir(test_folder + test_subfolder))
    start_index = 0
    last_index = batch_size
    while True:
        gc.collect()
        files_in_batch = []
        print('Evaluation with --> ' + train_folder + train_subfolder)
        for j in range(start_index, last_index):
            files_in_batch.append(test_folder + test_subfolder + f'/{j}.npz')
        iSeVCs, vulnLocMatrixes, labels = create_batch(files_in_batch)
        vuldee_model.evaluate([iSeVCs, vulnLocMatrixes], labels, batch_size=batch_size)
        del files_in_batch, iSeVCs, vulnLocMatrixes, labels
        start_index = last_index
        if start_index == test_files_lenght-1:
            break
        last_index += batch_size
        if last_index > (test_files_lenght - 1):
            last_index = test_files_lenght - 1


vuldee_model.save('/home/httpiego/PycharmProjects/VulDeeDiegator/trained_models/trained_model2.h5')

# Plot the results after training
plt.figure(figsize=(12, 6))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(all_losses, label='Training Loss')
plt.title('Training Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(all_accuracies, label='Training Accuracy')
plt.title('Training Accuracy Over Time')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

