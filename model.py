import tensorflow as tf
from tensorflow.keras import layers, models

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

    # Multiply layer (to highlight vulnerable tokens)
    multiply = layers.Multiply(name='multiply_layer')([activations, tf.expand_dims(vulnerability_location_matrix, -1)])

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

# Train the model
#vuldee_model.fit([iSeVCs, vulnerability_location_matrix], labels, epochs=5, batch_size=32)