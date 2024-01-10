# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define necessary functions
# Function to save files
def save_file(values, file_name, file_directory):
    
    # Save the file as a .npy file
    np.save(os.path.join(file_directory, file_name), values)
    
    print(f'Saved {file_name} to {file_directory}')


# CNN class
class CNN():

    def __init__(self, input_shape, output_shape):
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        # Initialize input_layer here
        self.input_layer = None  

    # Method to build the hidden layers
    def build_hidden_layers(self):
        
        # Convolutional Layers
        # First Convolutional Layer
        x1 = tf.keras.layers.Conv1D(filters=1000, kernel_size=1, padding='same', activation='LeakyReLU')(self.input_layer)
        x1 = tf.keras.layers.BatchNormalization()(x1)

        # Second Convolutional Layer
        x2 = tf.keras.layers.Conv1D(filters=1000, kernel_size=1, padding='same', activation='LeakyReLU')(x1)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        
        # Third Convolutional Layer
        x3 = tf.keras.layers.Conv1D(filters=1000, kernel_size=1, padding='same', activation='LeakyReLU')(x2)
        x3 = tf.keras.layers.BatchNormalization()(x3)

        return x3

    # Method to build the overall model
    def build_model(self):
        
        # Input layer
        self.input_layer = tf.keras.layers.Input(shape=self.input_shape)

        # Hidden layer
        hidden_layer = self.build_hidden_layers()

        # Output Layer
        output_layer = tf.keras.layers.Conv1D(filters=self.output_shape[1], kernel_size=1, padding='same', activation='LeakyReLU')(hidden_layer)

        # Build model
        self.model = tf.keras.models.Model(inputs=[self.input_layer], outputs=[output_layer])

        return self.model

    # Method to compile the model
    def compile(self, optimizer, loss, evaluation_metric):
        
        # Compile model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=evaluation_metric)

        return self.model
    
    # Define method to train the model
    def train(self, x_train, y_train, epochs, batch_size):
        
        # Train model
        self.history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        
        return self.history
    
    # Method to print summary of model
    def summary(self):
        
        self.model.summary()


# Define main function
def main():

    # Clear screen
    os.system('clear')

    # Define directories
    current_directory = os.getcwd()

    # Define directory for the normalized data
    normalized_data_directory = os.path.join(current_directory, 'data', 'normalized')

    # Define directory for the trained results
    trained_results_directory = os.path.join(current_directory, 'cnn', 'training_results')

    # Working with just the displacement data to conserve memory
    # Load the normalized training subsets for displacement data
    print('Loading the normalized training subsets for displacement data...')
    normalized_training_displacement_data = np.load(os.path.join(normalized_data_directory, 'normalized_training_displacement_data.npy'))

    # Print the shapes of the displacement data
    print(f'The shape of displacement data is {normalized_training_displacement_data.shape[1:]}.')

    # Load the normalized training subsets for force data
    print('Loading the normalized training subsets for force data...')
    normalized_training_force_data = np.load(os.path.join(normalized_data_directory, 'normalized_training_force_data.npy'))

    # Print the shapes of the force data
    print(f'The shape of force data is {normalized_training_force_data.shape[1:]}.')

    # Define variables that remain constant during the training
    input_shape = normalized_training_displacement_data.shape[1:]
    output_shape = normalized_training_force_data.shape[1:]

    # Create an instance of the CNN class
    model = CNN(input_shape, output_shape)

    # Build and the model
    model.build_model()
    model.compile(optimizer = 'nadam', loss = 'mse', evaluation_metric = 'mae')

    # Print the model summary
    model.summary()


if __name__ == '__main__':
    
    # Call main function
    main()

    # Change fonts and specify font size
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    FONT_SIZE = 12