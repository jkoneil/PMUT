% This script is written to use the DAS function
% for beamforming
% with simulated data from kWave

clear all; close all; clc;

% load data
load('simulated_sensor_data_double_pitch_18MHz.mat');
channel_data = sensor_data';


% run DAS algorithm
rf = channel_data;
no_ele = size(rf,2); % number of transducer elements

% Fs = 100*1e6;
channelSpacing = pitch_mm; % [mm] % distance between elements
speedSound = 1540; % [m/s]
sampleSpacing = (1/fs*speedSound)*1000; % for PA data, how far sound travels per time sample
x_axis = [1:size(rf,2)] * channelSpacing; % [mm]
y_axis = [1:size(rf,1)] * sampleSpacing; % [mm]

f_num = 4; % aperture growth may not work well

bf_data = DAS(rf, no_ele, fs, channelSpacing, speedSound);

% visualize the channel data
subplot(1, 2, 1);
imagesc(x_axis, y_axis, channel_data);
xlabel('lateral [mm]');
ylabel('axial [mm]');
axis image;
colormap gray;
colorbar;
title('Before beamforming');

% visualize the data
bf_data_env = abs(hilbert(bf_data));
bf_data_norm = bf_data_env./max(bf_data_env(:));
bf_data_db = db(bf_data_norm);


subplot(1,2,2);
imagesc(x_axis, y_axis, bf_data_db,[-40,0]);
xlabel('lateral [mm]');
ylabel('axial [mm]');
axis image;
colormap gray;
colorbar;
title('After beamforming')