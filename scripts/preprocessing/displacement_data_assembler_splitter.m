clc; clear all;

% Get the current directory
current_directory = pwd;

% Define directory of the data
% raw_data_directory = fullfile(current_directory, '..', '..', 'data_generation');

% Define directory of the assembled data
assembled_data_directory = fullfile(current_directory, '..', 'data_assembled');

% List all the folders in the raw data directory
raw_data_folders = dir(raw_data_directory);

% Remove the first two folders (.) and (..)
raw_data_folders = raw_data_folders(3:end);
raw_data_folders = {raw_data_folders.name};

%% Data Stack
% Loop through all the folders
for n_sample = 1:length(raw_data_folders)
    
    % Print which sample is being processed
    fprintf('Processing sample %d of %d...\n', n_sample, length(raw_data_folders));
    
    % Define the folder name and full path
    folder_name = raw_data_folders{n_sample};
    folder_directory = fullfile(raw_data_directory, folder_name);
    
    if n_sample == 1
        % If first sample, set the data to a variable
        
        % Load the data
        displacement_data = load(fullfile(folder_directory, 'u_history.mat')).um_history;
        
        % Convert the data from cell to matrix
        displacement_data = cat(3, displacement_data{:});
        
    else
        % If any other sample, stack the data to existing data
        
        % Load the data from other samples
        additional_displacement_data = load(fullfile(folder_directory, 'u_history.mat')).um_history;
        
        % Convert the data from cell to matrix
        additional_displacement_data = cat(3, additional_displacement_data{:});
        
        % Stack the data
        displacement_data = cat(3, displacement_data, additional_displacement_data);
        
    end
    
end

%% Reshape
% Reshape the data such that the shape is: [n_samples, n_time_steps, n_sensors]
displacement_data = permute(displacement_data, [3, 2, 1]);

[num_samples, num_timesteps, num_sensors] = size(displacement_data);

%% Dataset Split
training_proportion = 0.8;
validation_proportion = 0.1;
test_proportion = 0.1;

training_indices = training_proportion * num_samples;
validation_indices = training_indices + validation_proportion * num_samples;

training_displacement_data = displacement_data(1:training_indices, :, :);
validation_displacement_data = displacement_data(training_indices:validation_indices-1, :, :);
test_displacement_data = displacement_data(validation_indices+1:end, :, :);

%% Save Data
% Save the data as a .npy file
fprintf('Saving data...\n');
writeNPY(training_displacement_data, fullfile(assembled_data_directory, 'training_displacement_data.npy'));
writeNPY(validation_displacement_data, fullfile(assembled_data_directory, 'validation_displacement_data.npy'));
writeNPY(test_displacement_data, fullfile(assembled_data_directory, 'test_displacement_data.npy'));