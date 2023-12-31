clc; clear all;

% Get the current directory
current_directory = pwd;

% Define directory of the data
% raw_data_directory = fullfile(current_directory, '..', '..', 'data', 'generation');

% Define directory of the assembled data
assembled_data_directory = fullfile(current_directory, '..', 'data', 'assembled');

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
        force_data = load(fullfile(folder_directory, 'F_history.mat')).F_eff_history;
        
    else
        % If any other sample, stack the data to existing data
        
        % Load the data from other samples
        additional_force_data = load(fullfile(folder_directory, 'F_history.mat')).F_eff_history;
        
        % Vertically stack the cells
        force_data = vertcat(force_data, additional_force_data);
        
    end
    
end

clear additional_force_data

%% Data Extraction
% Pre-define a matrix of (num_samples, num_timesteps, num_sensors)
num_samples = length(force_data);
[num_sensors, num_timesteps] = size(force_data{1});

extracted_force_data = zeros(num_samples, num_sensors, num_timesteps);

% Now individually go into each cell and then extract data
for n_cell = 1:length(force_data)
    
    % Extract the cell data
    individual_force_data = force_data{n_cell};
    
    % Convert sparse to regular matrix
    individual_force_data = full(individual_force_data);
    
    % Set the individual values to the predefined force data
    extracted_force_data(n_cell, :, :) = individual_force_data;
    
    % Print a message to see progress
    if mod(n_cell, 2500) == 0
        
        fprintf('Finished extracting data from %d samples...\n', n_cell)
        
    end
    
end

%% Reshape
% Reshape the data such that the shape is: [n_samples, n_time_steps, n_sensors]
extracted_force_data = permute(extracted_force_data, [1, 3, 2]);

[num_samples, num_timesteps, num_sensors] = size(force_data);

%% Dataset Split
training_proportion = 0.8;
validation_proportion = 0.1;
test_proportion = 0.1;

training_indices = training_proportion * num_samples;
validation_indices = training_indices + validation_proportion * num_samples;

training_force_data = extracted_force_data(1:training_indices, :, :);
validation_force_data = extracted_force_data(training_indices:validation_indices-1, :, :);
test_force_data = extracted_force_data(validation_indices+1:end, :, :);

%% Save Data
% Save the data as a .npy file
fprintf('Saving data...\n');
writeNPY(training_force_data, fullfile(assembled_data_directory, 'training_force_data.npy'));
writeNPY(validation_force_data, fullfile(assembled_data_directory, 'validation_force_data.npy'));
writeNPY(test_force_data, fullfile(assembled_data_directory, 'test_force_data.npy'));