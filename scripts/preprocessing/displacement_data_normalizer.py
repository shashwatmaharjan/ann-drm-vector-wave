# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt

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

    # Working with just the displacement data to conserve memory
    # Load the training subsets for displacement data
    print('Loading the training subsets for displacement data...')
    training_displacement_data = np.load(os.path.join(assembled_data_directory, 'training_displacement_data.npy'))

    # Print the shapes of the displacement data
    print(f'The shape of displacement data is {training_displacement_data.shape[1:]}.')

    # Get normalizing parameters for displacement
    displacement_mean = np.mean(training_displacement_data)
    displacement_range = np.max(training_displacement_data) - np.min(training_displacement_data)

    # Normalize the training displacement values
    print('Normalizing the training displacement data...')
    normalized_training_displacement_data = normalize_data(training_displacement_data, displacement_mean, displacement_range)

    # Clear the variables to free up memory
    del training_displacement_data

    # Save the normalized displacement data as a .npy file
    save_file(normalized_training_displacement_data, 'normalized_training_displacement_data.npy', normalized_data_directory)

    # Clear the variables to free up memory
    del normalized_training_displacement_data

    # Load the validation subsets for displacement data
    print('Loading the validation subsets for displacement data...')
    validation_displacement_data = np.load(os.path.join(assembled_data_directory, 'validation_displacement_data.npy'))

    # Normalize the validation displacement values
    print('Normalizing the validation displacement data...')
    normalized_validation_displacement_data = normalize_data(validation_displacement_data, displacement_mean, displacement_range)

    # Load the test subsets for displacement data
    print('Loading the test subsets for displacement data...')
    test_displacement_data = np.load(os.path.join(assembled_data_directory, 'test_displacement_data.npy'))

    # Normalize the test displacement values
    print('Normalizing the test displacement data...')
    normalized_test_displacement_data = normalize_data(test_displacement_data, displacement_mean, displacement_range)

    # Clear the variables to free up memory
    del validation_displacement_data, test_displacement_data

    # Save the normalized displacement data as a .npy file
    save_file(normalized_validation_displacement_data, 'normalized_validation_displacement_data.npy', normalized_data_directory)
    save_file(normalized_test_displacement_data, 'normalized_test_displacement_data.npy', normalized_data_directory)

    # Save the normalizing parameters as a .npy file
    # Bundle everything into a single dictionary
    normalizing_parameters = {'displacement_mean': displacement_mean,
                            'displacement_range': displacement_range}

    # Save the normalizing parameters as a .npy file
    save_file(normalizing_parameters, 'normalizing_displacement_parameters.npy', normalized_data_directory)


if __name__ == '__main__':
    
    # Call main function
    main()

    # Change fonts and specify font size
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    FONT_SIZE = 12