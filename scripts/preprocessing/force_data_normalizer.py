# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Define necessary functions
# Define function to normalize the force dataset
def normalize_data(data_to_be_normalized, training_mean, training_range):

    normalized_data = (data_to_be_normalized - training_mean) / training_range

    return normalized_data


# Function to save files
def save_file(values, file_name, file_directory):
    
    # Save the file as a .npy file
    np.save(os.path.join(file_directory, file_name), values)
    
    print(f'Saved {file_name} to {file_directory}')


# Define main function
def main():

    # Clear screen
    os.system('clear')

    # Define directories
    current_directory = os.getcwd()

    # Define directory for the assembled data
    assembled_data_directory = os.path.join(current_directory, 'data', 'assembled')

    # Define directory for the normalized data
    normalized_data_directory = os.path.join(current_directory, 'data', 'normalized')

    # Working with just the force data to conserve memory
    # Load the training subsets for force data
    print('Loading the training subsets for force data...')
    training_force_data = np.load(os.path.join(assembled_data_directory, 'training_force_data.npy'))

    # Print the shapes of the force data
    print(f'The shape of force data is {training_force_data.shape[1:]}.')

    # Get normalizing parameters for force
    force_mean = np.mean(training_force_data)
    force_range = np.max(training_force_data) - np.min(training_force_data)

    # Normalize the training force values
    print('Normalizing the training force data...')
    normalized_training_force_data = normalize_data(training_force_data, force_mean, force_range)

    # Clear the variables to free up memory
    del training_force_data

    # Save the normalized force data as a .npy file
    save_file(normalized_training_force_data, 'normalized_training_force_data.npy', normalized_data_directory)

    # Clear the variables to free up memory
    del normalized_training_force_data

    # Load the validation subsets for force data
    print('Loading the validation subsets for force data...')
    validation_force_data = np.load(os.path.join(assembled_data_directory, 'validation_force_data.npy'))

    # Normalize the validation force values
    print('Normalizing the validation force data...')
    normalized_validation_force_data = normalize_data(validation_force_data, force_mean, force_range)

    # Load the test subsets for force data
    print('Loading the test subsets for force data...')
    test_force_data = np.load(os.path.join(assembled_data_directory, 'test_force_data.npy'))

    # Normalize the test force values
    print('Normalizing the test force data...')
    normalized_test_force_data = normalize_data(test_force_data, force_mean, force_range)

    # Clear the variables to free up memory
    del validation_force_data, test_force_data

    # Save the normalized force data as a .npy file
    save_file(normalized_validation_force_data, 'normalized_validation_force_data.npy', normalized_data_directory)
    save_file(normalized_test_force_data, 'normalized_test_force_data.npy', normalized_data_directory)

    # Save the normalizing parameters as a .npy file
    # Bundle everything into a single dictionary
    normalizing_parameters = {'force_mean': force_mean,
                            'force_range': force_range}

    # Save the normalizing parameters as a .npy file
    save_file(normalizing_parameters, 'normalizing_force_parameters.npy', normalized_data_directory)


if __name__ == '__main__':
    
    # Call main function
    main()

    # Change fonts and specify font size
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    FONT_SIZE = 12