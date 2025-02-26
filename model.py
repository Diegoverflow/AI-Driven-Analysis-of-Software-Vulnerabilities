import gc

import tensorflow as tf
import keras.api.metrics as m
import tensorflow.keras.metrics as mt
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import backend as K
from keras.api.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from keras.api.callbacks import EarlyStopping
from tensorflow.keras import layers, models
import os
import numpy as np
import matplotlib.pyplot as plt
import random

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected.")

sequence_length = 2618
embedding_dim = 32
hidden_nodes = 32  #900
hidden_layers = 2  #2
output_dim = 512
k_value = 1 
num_classes = 1
batch_size = 128  #16
learning_rate = 0.002
dropout_rate = 0.4
epochs = 10



def build_vuldee_model(sequence_length, embedding_dim, hidden_nodes, hidden_layers, k_value, num_classes, dropout_rate):
    inputs = layers.Input(shape=(sequence_length, embedding_dim), name='input_layer')

    temp = inputs
    for i in range(hidden_layers):
        layer = layers.Bidirectional(
            layers.GRU(units=hidden_nodes,
                       return_sequences=True,
                       #kernel_regularizer=regularizers.l2(0.01),
                       #recurrent_regularizer=regularizers.l2(0.01)
                       ),
            name=f'bidirectional_rnn_layer_{i + 1}')(temp)
        temp = layer

    dense = layers.Dense(hidden_nodes,
                         activation='relu',
                         #kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                         name='dense_layer')(temp)

    #dropout = layers.Dropout(dropout_rate, name='dropout_layer')(dense)
    #batch_norm = BatchNormalization(name='batch_normalization_layer')(dropout)
    activations = layers.Activation('relu', name='activation_layer')(dense)

    vulnerability_location_matrix = layers.Input(shape=(sequence_length,), name='vulnerability_location_input')
    expanded_vulnerability_location_matrix = layers.Lambda(lambda x: tf.expand_dims(x, -1))(
        vulnerability_location_matrix)

    multiply = layers.Multiply(name='multiply_layer')([activations, expanded_vulnerability_location_matrix])
    k_max_values = layers.Lambda(lambda x: tf.math.top_k(x, k=k_value).values,
                                 #output_shape=(k_value, embedding_dim),
                                 output_shape=(sequence_length, k_value),
                                 name='k_max_pooling')(multiply)

    average_pooling = layers.GlobalAveragePooling1D(name='average_pooling_layer')(k_max_values)
    #reduction = layers.Dense(output_dim, activation='relu', name='reduction_layer')(average_pooling)
    #output = layers.Dense(num_classes, activation='sigmoid', name='output_layer')(average_pooling)

    model = models.Model(inputs=[inputs, vulnerability_location_matrix], outputs=average_pooling, name='VulDeeLocator_NN')
    return model


vuldee_model = build_vuldee_model(sequence_length, embedding_dim, hidden_nodes, hidden_layers, k_value, num_classes,
                                  dropout_rate)

vuldee_model.compile(optimizer=Adamax(learning_rate=learning_rate), loss='binary_crossentropy',
                     metrics=['accuracy', m.Precision, m.Recall, m.F1Score])
vuldee_model.summary()

train_folder = '/home/httpiego/PycharmProjects/AI-Analysis/iSeVCs/Vectorized/' + str(embedding_dim) + '/Training/'
test_folder = '/home/httpiego/PycharmProjects/AI-Analysis/iSeVCs/Vectorized/' + str(embedding_dim) + '/Testing/'


def plot_train(losses, accuracies, train_num, plot_num):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss', color='blue')
    plt.title('Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Training Accuracy', color='blue')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f'/home/httpiego/PycharmProjects/AI-Analysis/trained_models/{train_num}/train_{plot_num}.png')

def plot_test(losses, accuracies, train_num, plot_num):
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Testing Loss', color='orange')
    plt.title('Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Testing Accuracy', color='orange')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f'/home/httpiego/PycharmProjects/AI-Analysis/trained_models/{train_num}/test_{plot_num}.png')


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
    data = np.load(test_folder + file_name)  #, allow_pickle=True)

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


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
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

for i in range(epochs):
    all_losses = []
    all_accuracies = []
    all_f1s = []
    all_precisions = []
    all_recalls = []
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
        history = vuldee_model.fit([iSeVCs, vulnLocMatrixes], labels,
                                   epochs=1, batch_size=batch_size,
                                   #callbacks=[early_stopping],
                                   #callbacks=[early_stopping, reduce_lr],
                                   #validation_split=0.2
                                   )

        all_losses.append(history.history['loss'])
        all_accuracies.append(history.history['accuracy'])
        all_f1s.append(history.history['f1_score'])
        all_recalls.append(history.history['recall'])
        all_precisions.append(history.history['precision'])

        del files_in_batch, iSeVCs, vulnLocMatrixes, labels
        gc.collect()
        start_index = last_index
        last_index += batch_size
        if last_index > train_files_lenght:
            last_index = train_files_lenght

    #EVALUATE

    print('TEST')

    all_test_losses = []
    all_test_accuracies = []
    all_test_f1s = []
    all_test_recalls = []
    all_test_precisions = []

    start_index = 0
    last_index = batch_size
    while True:
        gc.collect()
        files_in_batch = []
        for j in range(start_index, last_index):
            files_in_batch.append(test_files[j])
        iSeVCs, vulnLocMatrixes, labels = create_batch_test(files_in_batch)
        results = vuldee_model.evaluate([iSeVCs, vulnLocMatrixes], labels, batch_size=batch_size)

        print(results)

        all_test_losses.append(results[0])
        all_test_accuracies.append(results[1])
        all_test_precisions.append(results[2])
        all_test_recalls.append(results[3])
        all_test_f1s.append(results[4])


        del files_in_batch, iSeVCs, vulnLocMatrixes, labels
        gc.collect()
        start_index = last_index
        if start_index == test_files_lenght - 1:
            average_loss = np.mean(all_test_losses)
            average_accuracy = np.mean(all_test_accuracies)
            average_prec = np.mean(all_test_precisions)
            average_recalls = np.mean(all_test_recalls)
            average_f1s = np.mean(all_test_f1s)

            print('==========================================================')
            print('======================= EVALUATE =========================')
            print('==========================================================')
            print(f'Average Test Loss: {average_loss:.4f}')
            print(f'Average Test Accuracy: {average_accuracy:.4f}')
            print(f'Average Test Precision: {average_prec:.4f}')
            print(f'Average Test Recall: {average_recalls:.4f}')
            print(f'Average Test F1-Scor: {average_f1s:.4f}')
            print('==========================================================')
            print('==========================================================')

            break
        last_index += batch_size
        if last_index > (test_files_lenght - 1):
            last_index = test_files_lenght - 1

        average_loss = np.mean(all_test_losses)
        average_accuracy = np.mean(all_test_accuracies)
        average_prec = np.mean(all_test_precisions)
        average_recalls = np.mean(all_test_recalls)
        average_f1s = np.mean(all_test_f1s)

        print('==========================================================')
        print('======================= EVALUATE =========================')
        print('==========================================================')
        print(f'Average Test Loss: {average_loss:.4f}')
        print(f'Average Test Accuracy: {average_accuracy:.4f}')
        print(f'Average Test Precision: {average_prec:.4f}')
        print(f'Average Test Recall: {average_recalls:.4f}')
        print(f'Average Test F1-Scor: {average_f1s:.4f}')
        print('==========================================================')
        print('==========================================================')
        print('==========================================================')

    average_train_loss = np.mean(all_losses)
    average_train_accuracy = np.mean(all_accuracies)
    average_train_prec = np.mean(all_precisions)
    average_train_recalls = np.mean(all_recalls)
    average_train_f1s = np.mean(all_f1s)
    print('==========================================================')
    print('======================= TRAINING =========================')
    print('==========================================================')
    print(f'Average Train Loss: {average_train_loss:.4f}')
    print(f'Average Train Accuracy: {average_train_accuracy:.4f}')
    print(f'Average Train Precision: {average_train_prec:.4f}')
    print(f'Average Train Recall: {average_train_recalls:.4f}')
    print(f'Average Train F1-Scor: {average_f1s:.4f}')
    print('==========================================================')
    print('==========================================================')
    print('==========================================================')

    # Plot the results after training
    plot_train(all_losses, all_accuracies, train_num, i)
    plot_test(all_test_losses, all_test_accuracies, train_num, i)

vuldee_model.save(
    f'/home/httpiego/PycharmProjects/AI-Analysis/trained_models/{train_num}/trained_model_{train_num}.keras')
