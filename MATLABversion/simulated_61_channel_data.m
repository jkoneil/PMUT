% This script is written to simulate 61 channel data collected by PMUT
% array, outputting multiplexed records (1024 x 16 x 4)

% Initiate a 2D matrix
% vertical axia: depth
% horizontal axis: channel

num_samples = 1024;
num_channels = 61;
all_data = zeros(num_samples,num_channels);

% PMUT information
f_c = 10e6; % [Hz]
f_s = 4 * f_c; % 200% Nyquist
s_o_s = 1480e3; % [mm/s]
sample_spacing = 1/f_s*s_o_s; %[mm]

% Simulate point targets at 0.5, 1, 2, 5, 10 mm
target_depths_mm = [0.5, 1, 2, 5, 10]; % [mm]
target_depths_pixel = round(target_depths_mm/sample_spacing); % in pixel
int_curve_hor = [1, 0.7, 0.3]; % for each layer from the center
int_curve_ver = [0.1, 0.3, 0.7, 0.9, 1, 0.9, 0.7, 0.3, 0.1];

% index array of selected elements
element_array = [14, 15, 16, 21, 22, 23, 24, 29, 30, 31, 32, 33, 38, 39, ...
    40, 41, 46, 47, 48];
third_layer = [14, 15, 16, 21, 24, 29, 33, 38, 41, 46, 47, 48];
second_layer = [22, 23, 30, 32, 39, 40];
first_layer = [31];

% populate non-zero signals
% all_data(:,element_array) = 1;

% render in the vertical direction
for i = 1:size(target_depths_pixel,2)
    target_depth = target_depths_pixel(i);
    neighbor_size = (size(int_curve_ver,2) - 1)/2;
    target_neighbor = [(target_depth - neighbor_size):(target_depth + neighbor_size)];

    for j = 1:size(element_array,2)
        all_data(target_neighbor,element_array(j)) = 1;
        all_data(target_neighbor,element_array(j)) = all_data(target_neighbor,element_array(j)).*int_curve_ver';
    end

end

% render in the horizontal direction
all_data(:,first_layer) = all_data(:,first_layer) * int_curve_hor(1);
all_data(:,second_layer) = all_data(:,second_layer) * int_curve_hor(2);
all_data(:,third_layer) = all_data(:,third_layer) * int_curve_hor(3);

%% Convert full 61-channel data into multiplexed form with output size: 1024 x 16 x 4

multiplexed_data = zeros(num_samples, 16, 4);

for rec = 1:16
    for sec = 1:4
        ch = (sec-1)*16 + rec;
        if ch <= 61
            multiplexed_data(:, rec, sec) = all_data(:, ch);
        end
    end
end

% save data
save('data_visualization.mat','multiplexed_data');
