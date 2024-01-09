# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define necessary functions
# Define function to normalize the displacement and force dataset
def normalize_data(data_to_be_normalized, training_mean, training_range):

    normalized_data = (data_to_be_normalized - training_mean) / training_range

    return normalized_data

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

    # Define directories
    current_directory = os.getcwd()

    # Define directory for the assembled data
    assembled_data_directory = os.path.join(current_directory, 'data', 'assembled')

    # Define directory for the trained results
    trained_results_directory = os.path.join(current_directory, 'cnn', 'training_results')

    # Working with just the displacement data to conserve memory
    # Load the training subsets for displacement data
    training_displacement_data = np.load(os.path.join(assembled_data_directory, 'training_displacement_data.npy'))

    # Print the shapes of the displacement data
    print(f'The shape of displacement data is {training_displacement_data.shape[1:]}.')

    # Get normalizing parameters for displacement
    displacement_mean = np.mean(training_displacement_data)
    displacement_range = np.max(training_displacement_data) - np.min(training_displacement_data)

    # Normalize the displacement values
    print('Normalizing the displacement data...')
    normalized_training_displacement_data = normalize_data(training_displacement_data, displacement_mean, displacement_range)

    # Clear the variables to free up memory
    del training_displacement_data

    # Load the training subsets for force data
    training_force_data = np.load(os.path.join(assembled_data_directory, 'training_force_data.npy'))

    # Print the shapes of the force data
    print(f'The shape of force data is {training_force_data.shape[1:]}.')

    # Get normalizing parameters for force
    force_mean = np.mean(training_force_data)
    force_range = np.max(training_force_data) - np.min(training_force_data)

    # Normalize the force values
    print('Normalizing the force data...')
    normalized_training_force_data = normalize_data(training_force_data, force_mean, force_range)
    
    # Clear the variables to free up memory
    del training_force_data

    # Save the normalizing parameters as a .npy file
    # Bundle everything into a single dictionary
    normalizing_parameters = {'displacement_mean': displacement_mean,
                            'displacement_range': displacement_range,
                            'force_mean': force_mean,
                            'force_range': force_range}

    # Save the normalizing parameters as a .npy file
    save_file(normalizing_parameters, 'normalizing_parameters.npy', trained_results_directory)

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