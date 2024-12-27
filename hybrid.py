import gc

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.api.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from keras.api.callbacks import EarlyStopping
from tensorflow.keras import layers, models
import os
import numpy as np
import matplotlib.pyplot as plt
import random

# Hyperparameters
sequence_length = 2618  #295 average / #2618 longest
embedding_dim = 64
rnn_units = 32
k_value = 5
num_classes = 1
batch_size = 128  # 500

cnn_filters = 128
kernel_size = 3
rhn_units = 32 ###########
depth = 3  # Number of RHN layers


def build_hybrid_cnn_rhn_model(sequence_length, embedding_dim, cnn_filters, kernel_size, rhn_units, depth, k_value, num_classes):
    # Input layer
    inputs = layers.Input(shape=(sequence_length, embedding_dim), name='input_layer')

    # Convolutional layer
    cnn_output = layers.Conv1D(
        filters=cnn_filters, kernel_size=kernel_size, activation='relu',
        padding='same', kernel_regularizer=regularizers.l2(0.01),
        name='cnn_layer')(inputs)

    # Max pooling to reduce sequence length
    pooled_output = layers.MaxPooling1D(pool_size=2, name='max_pooling_layer')(cnn_output)

    # Projection layer to match RHN input size
    projected_output = layers.Dense(rhn_units, activation='relu', name='projection_layer')(pooled_output)

    # Recurrent Highway Network
    h = projected_output
    for i in range(depth):
        transform_gate = layers.Dense(
            rhn_units, activation='sigmoid', name=f'transform_gate_{i}',
            kernel_regularizer=regularizers.l2(0.01))(h)
        carry_gate = layers.Lambda(lambda x: 1.0 - x, name=f'carry_gate_{i}')(transform_gate)

        highway_h = layers.Dense(
            rhn_units, activation='relu', name=f'highway_dense_{i}',
            kernel_regularizer=regularizers.l2(0.01))(h)

        h = layers.Add(name=f'highway_add_{i}')([
            layers.Multiply(name=f'transform_gate_apply_{i}')([transform_gate, highway_h]),
            layers.Multiply(name=f'carry_gate_apply_{i}')([carry_gate, h])
        ])

    # Dense layer after RHN
    dense = layers.Dense(
        rhn_units, activation='relu',
        kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
        name='dense_layer')(h)

    # Dropout for regularization
    dropout = layers.Dropout(0.5, name='dropout_layer')(dense)

    # Vulnerability location matrix processing
    vulnerability_location_matrix = layers.Input(shape=(sequence_length,), name='vulnerability_location_input')
    expanded_vulnerability_location_matrix = layers.Lambda(lambda x: tf.expand_dims(x, -1), name='expand_dims')(vulnerability_location_matrix)

    # Align sequence length of vulnerability location matrix with the dropout output
    aligned_vulnerability_location_matrix = layers.Lambda(
        lambda x: tf.image.resize(x, [tf.shape(dropout)[1], 1]),
        name='resize_vulnerability_location')(expanded_vulnerability_location_matrix)

    # Perform element-wise multiplication
    multiply = layers.Multiply(name='multiply_layer')([dropout, aligned_vulnerability_location_matrix])

    # k-max pooling
    k_max_values = layers.Lambda(lambda x: tf.math.top_k(x, k=k_value).values, name='k_max_pooling')(multiply)

    # Average pooling
    average_pooling = layers.GlobalAveragePooling1D(name='average_pooling_layer')(k_max_values)

    # Output layer
    output = layers.Dense(num_classes, activation='sigmoid', name='output_layer')(average_pooling)

    # Create the model
    model = models.Model(inputs=[inputs, vulnerability_location_matrix], outputs=output, name='Hybrid_CNN_RHN')

    return model


vuldee_model =  build_hybrid_cnn_rhn_model(sequence_length, embedding_dim, cnn_filters, kernel_size, rhn_units, depth, k_value, num_classes)


#vuldee_model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])
vuldee_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


vuldee_model.summary()


train_folder = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs/Vectorized/Training/'
test_folder = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs/Vectorized/Testing/'


def plot(all_losses, all_accuracies,  all_test_losses, all_test_accuracies, train_num, plot_num):
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(all_losses, label='Training Loss', color='blue')
    plt.plot(all_test_losses, label='Testing Loss', color='orange')
    plt.title('Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(all_accuracies, label='Training Accuracy', color='blue')
    plt.plot(all_test_accuracies, label='Testing Accuracy', color='orange')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    plt.savefig(f'/home/httpiego/PycharmProjects/VulDeeDiegator/trained_models/{train_num}/training_progress_{plot_num}.png')

def load_data_train(file_name):
    data = np.load(train_folder + file_name, allow_pickle=True)

    iSeVC = data['iSeVC']
    vulnLocMatrix = data['vulnLocMatrix']
    label = data['label']

    return iSeVC, vulnLocMatrix, label

def create_batch_train(file_list):
    iSeVCs = []
    vulnLocMatrixes = []
    labels = []

    for file_name in file_list:
        iSeVC, vulnLocMatrix, label = load_data_train(file_name)
        #iSeVC = np.squeeze(iSeVC, axis=1)
        iSeVCs.append(iSeVC)
        #vulnLocMatrix = np.squeeze(vulnLocMatrix, axis=1)
        vulnLocMatrixes.append(vulnLocMatrix)
        labels.append(label)

    iSeVCs = np.array(iSeVCs)
    iSeVCs = np.squeeze(iSeVCs, axis=1)
    vulnLocMatrixes = np.array(vulnLocMatrixes)
    vulnLocMatrixes = np.squeeze(vulnLocMatrixes, axis=1)
    labels = np.array(labels)

    return iSeVCs, vulnLocMatrixes, labels

def load_data_test(file_name):
    data = np.load(test_folder + file_name)#, allow_pickle=True)

    iSeVC = data['iSeVC']
    vulnLocMatrix = data['vulnLocMatrix']
    label = data['label']

    return iSeVC, vulnLocMatrix, label

def create_batch_test(file_list):
    iSeVCs = []
    vulnLocMatrixes = []
    labels = []

    for file_name in file_list:
        iSeVC, vulnLocMatrix, label = load_data_test(file_name)
        #iSeVC = np.squeeze(iSeVC, axis=1)
        iSeVCs.append(iSeVC)
        #vulnLocMatrix = np.squeeze(vulnLocMatrix, axis=1)
        vulnLocMatrixes.append(vulnLocMatrix)
        labels.append(label)

    iSeVCs = np.array(iSeVCs)
    iSeVCs = np.squeeze(iSeVCs, axis=1)
    vulnLocMatrixes = np.array(vulnLocMatrixes)
    vulnLocMatrixes = np.squeeze(vulnLocMatrixes, axis=1)
    labels = np.array(labels)

    return iSeVCs, vulnLocMatrixes, labels

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5, # 8 with reduce_lr #5 without
                               restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7)


train_files_lenght = len(os.listdir(train_folder))
train_files = os.listdir(train_folder)
random.shuffle(train_files)

test_files_lenght = len(os.listdir(test_folder))
test_files = os.listdir(test_folder)
random.shuffle(test_files)

train_num = 1

newpath = f'/home/httpiego/PycharmProjects/VulDeeDiegator/trained_models/{train_num}'
if not os.path.exists(newpath):
    os.makedirs(newpath)

#TRAIN

for i in range(10):
    all_losses = []
    all_accuracies = []
    start_index = 0
    last_index = batch_size
    while True:
        if start_index == train_files_lenght:
            check = False
            break
        files_in_batch = []
        print(f'batch: {start_index} - {last_index}')
        for j in range(start_index, last_index):
            files_in_batch.append(train_files[j])
        iSeVCs, vulnLocMatrixes, labels = create_batch_train(files_in_batch)

            #PROVA 2000/4000 EPOCHE - TOGLIERE EARLY STOPPING - TRAINING E VALIDAZIONE CLASSICHE
        tf.keras.backend.clear_session()
        print(f'start index - {start_index}')
        print(f'last index - {last_index}')
        history = vuldee_model.fit([iSeVCs, vulnLocMatrixes], labels,
                                       epochs=1, batch_size=batch_size,
                                       #callbacks=[early_stopping],
                                       #callbacks=[early_stopping, reduce_lr],
                                       #validation_split=0.2
                                       )

        all_losses.extend(history.history['loss'])
        all_accuracies.extend(history.history['accuracy'])

        del files_in_batch, iSeVCs, vulnLocMatrixes, labels
        gc.collect()
        start_index = last_index
        last_index += batch_size
        if last_index > train_files_lenght:
            last_index = train_files_lenght
    print('last for loop index --> ' + str(i))


    #EVALUATE

    all_test_losses = []
    all_test_accuracies = []

    start_index = 0
    last_index = batch_size
    while True:
        gc.collect()
        files_in_batch = []
        print('Evaluation with --> ' + test_folder)
        for j in range(start_index, last_index):
            files_in_batch.append(test_files[j])
        iSeVCs, vulnLocMatrixes, labels = create_batch_test(files_in_batch)
        loss, accuracy = vuldee_model.evaluate([iSeVCs, vulnLocMatrixes], labels, batch_size=batch_size)

        all_test_losses.append(loss)
        all_test_accuracies.append(accuracy)

        del files_in_batch, iSeVCs, vulnLocMatrixes, labels
        gc.collect()
        start_index = last_index
        if start_index == test_files_lenght-1:

            average_loss = np.mean(all_test_losses)
            average_accuracy = np.mean(all_test_accuracies)

            print('==========================================================')
            print('======================= EVALUATE =========================')
            print('==========================================================')
            print(f'Average Test Loss: {average_loss:.4f}')
            print(f'Average Test Accuracy: {average_accuracy:.4f}')
            print('==========================================================')
            print('==========================================================')
            print('==========================================================')

            break
        last_index += batch_size
        if last_index > (test_files_lenght - 1):
            last_index = test_files_lenght - 1

        average_loss = np.mean(all_test_losses)
        average_accuracy = np.mean(all_test_accuracies)

        print('==========================================================')
        print('======================= EVALUATE =========================')
        print('==========================================================')
        print(f'Average Test Loss: {average_loss:.4f}')
        print(f'Average Test Accuracy: {average_accuracy:.4f}')
        print('==========================================================')
        print('==========================================================')
        print('==========================================================')


        # Plot the results after training
    plot(all_losses, all_accuracies, all_test_losses, all_test_accuracies, train_num, i)

vuldee_model.save(f'/home/httpiego/PycharmProjects/VulDeeDiegator/trained_models/{train_num}/trained_model_{train_num}.keras')

