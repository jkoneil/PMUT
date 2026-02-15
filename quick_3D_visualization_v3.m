% This script is used to visualize the 3D data collected by hexagon
% transducer array
% extend this visualization into full 3D scale, 
% using simulated 3D data

% Y. Tang

clear all; clc;

% load the array information
load('data_visualization.mat') % simulated data
load('hex61_pitch80.mat');
locations_x_y = coords;

% plot out all coordinates on a 2D plane
x = coords(:,1)';
y = coords(:,2)';

%{
figure
scatter(x,y,36,'filled');
grid on
axis equal tight
xlabel('x');ylabel('y');title('Points');
%}

%% index of transducers for each layer
layer_5 = [1 2 3 4 5 6 11 12 18 19 26 27 35 ...
    36 43 44 50 51 56 57 58 59 60 61];
layer_4 = [7 8 9 10 13 17 20 25 28 34 37 42 ...
    45 49 52 53 54 55];
layer_3 = [14 15 16 21 24 29 33 38 41 46 47 48];
layer_2 = [22 23 30 32 39 40];
layer_1 = [31];


%% simualate the signal intensity obtained by all elements
% this part can be replaced with Kwave simulation.
%{
int = zeros(1,61);
int(layer_1) = 0.6;
int(layer_2) = 0.7;
int(layer_3) = 0.8;
int(layer_4) = 0.9;
int(layer_5) = 1.0;
%}

%% Parameters
num_samples = size(all_data, 1); % 1024
%grid_range_x = -400:4:400; %-400:1:400;
%grid_range_x_y = -400:4:400; %-400:1:400;
% Crop grid to only cover the hexagon
%margin = 10; % small padding
grid_range_x = min(x) : 4 : max(x);
grid_range_y = min(y) : 4 : max(y);
% Create a grid for the entire image
[grid_x, grid_y] = meshgrid(grid_range_x, grid_range_y);  % cropped grid
% Allocate 3D volume
volume_3d = zeros(length(grid_range_y), length(grid_range_x), num_samples);

%% Loop through all depths
for depth_idx = 1:num_samples
    int = all_data(depth_idx, :); % 1x61 slice
    %% interpolation algorithm
    known_x = [x];
    known_y = [y];
    known_intensity = [int];
    % interpolate to the 2D grid
    slice = griddata(known_x, known_y, known_intensity, grid_x, grid_y, 'cubic');
    slice(isnan(slice)) = 0;   % replace NaNs with 0
    volume_3d(:, :, depth_idx) = slice;
end


%% Apply simple depth-only smoothing
sigma_xy = 1e-6;    % almost no blur in x/y
sigma_z = 5;       % blur along depth
volume_3d_smoothed = imgaussfilt3(volume_3d, [sigma_xy sigma_xy sigma_z]);
% normalize the smoothed volume
volume_3d_smoothed = volume_3d_smoothed / max(volume_3d_smoothed(:));

% Plot the 3D interpolated images
h = volshow(volume_3d_smoothed, 'Colormap', hot);
h.Alphamap = linspace(0,1,256).^1.72;  % min opacityf  = 0.
%title('3D Visualization');

%{
imagesc(axis_x,axis_y,interpolated_intensity);
colormap hot;
colorbar;
title('Interpolated Intensity Image');
axis equal tight;
%}

%% simulate for 3D imaging
% image_2d = interpolated_intensity;
% scaling_factors = [linspace(0.1, 1, 10),flip(linspace(0.1, 1, 10))];  % 10 scaling factors from 0.1 to 1
% image_3d = zeros([201, 201, 10]);
% 
% for i = 1:20
%     image_3d(:,:,i) = image_2d * scaling_factors(i);
% end

% volshow(image_3d);
% volshow(image_3d);
% title('Stacked Image Layers with Scaling Factors');