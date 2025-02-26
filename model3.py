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

sequence_length = 2618 
embedding_dim = 64
rnn_units = 32
k_value = 1
num_classes = 1
batch_size = 128 


def build_vuldee_model(sequence_length, embedding_dim, rnn_units, k_value, num_classes):

    inputs = layers.Input(shape=(sequence_length, embedding_dim), name='input_layer')
    #
    brnn = layers.Bidirectional(layers.LSTM(rnn_units,
                                            return_sequences=True,
                                            kernel_regularizer=regularizers.l2(0.01),
                                            recurrent_regularizer=regularizers.l2(0.01), ),
                                            name='bidirectional_rnn')(inputs)
    #
    #gru = layers.Bidirectional(layers.GRU(rnn_units, return_sequences=True, kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01),), name='bidirectional_rnn')(inputs)
    #

    dense = layers.Dense(rnn_units, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), activation='relu', name='dense_layer')(brnn)
    #
    #dense = layers.Dense(rnn_units, kernel_regularizer=regularizers.l2(0.001), activation='relu', name='dense_layer')(gru)
    #
    dropout = layers.Dropout(0.5)(dense) #
    #
    batch = BatchNormalization()(dropout)  #
    #
    activations = layers.Activation('relu', name='activation_layer')(batch)  #
    #
    vulnerability_location_matrix = layers.Input(shape=(sequence_length,), name='vulnerability_location_input')
    #
    expanded_vulnerability_location_matrix = layers.Lambda(lambda x: tf.expand_dims(x, -1))(
         vulnerability_location_matrix)
    #
    multiply = layers.Multiply(name='multiply_layer')([activations, expanded_vulnerability_location_matrix])
    #
    k_max_values = layers.Lambda(lambda x: tf.math.top_k(x, k=k_value).values, name='k_max_pooling')(multiply)
    #
    average_pooling = layers.GlobalAveragePooling1D(name='average_pooling_layer')(k_max_values)
    #
    output = layers.Dense(num_classes, activation='sigmoid', name='output_layer')(average_pooling)
    #
    model = models.Model(inputs=[inputs, vulnerability_location_matrix], outputs=output, name='VulDeeLocator_NN')
    #
    return model


vuldee_model = build_vuldee_model(sequence_length, embedding_dim, rnn_units, k_value, num_classes)

#vuldee_model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='binary_crossentropy', metrics=['accuracy'])
vuldee_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


vuldee_model.summary()


train_folder = '/home/httpiego/PycharmProjects/AI-Analysis/iSeVCs/Vectorized/Training/'
test_folder = '/home/httpiego/PycharmProjects/AI-Analysis/iSeVCs/Vectorized/Testing/'


def plot(all_losses, all_accuracies,  all_test_losses, all_test_accuracies, train_num, plot_num):
    plt.figure(figsize=(12, 6))

    
    plt.subplot(1, 2, 1)
    plt.plot(all_losses, label='Training Loss', color='blue')
    plt.plot(all_test_losses, label='Testing Loss', color='orange')
    plt.title('Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    
    plt.subplot(1, 2, 2)
    plt.plot(all_accuracies, label='Training Accuracy', color='blue')
    plt.plot(all_test_accuracies, label='Testing Accuracy', color='orange')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    plt.savefig(f'/home/httpiego/PycharmProjects/AI-Analysis/trained_models/{train_num}/training_progress_{plot_num}.png')

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

newpath = f'/home/httpiego/PycharmProjects/AI-Analysis/trained_models/{train_num}'
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


    plot(all_losses, all_accuracies, all_test_losses, all_test_accuracies, train_num, i)

vuldee_model.save(f'/home/httpiego/PycharmProjects/AI-Analysis/trained_models/{train_num}/trained_model_{train_num}.keras')

