% This script is used to visualize the 3D data collected by hexagon
% transducer array
% extend this visualization into full 3D scale, 
% using simulated 3D data

% Y. Tang

clear all; clc;

% load the array information
load('hex61_pitch80.mat');
locations_x_y = coords;

% plot out all coordinates on a 2D plane
x = coords(:,1)';
y = coords(:,2)';

figure
scatter(x,y,36,'filled');
grid on
axis equal tight
xlabel('x');ylabel('y');title('Points');

%% index of transducers for each layer
layer_1 = [1 2 3 4 5 6 11 12 18 19 26 27 35 ...
    36 43 44 50 51 56 57 58 59 60 61];
layer_2 = [7 8 9 10 13 17 20 25 28 34 37 42 ...
    45 49 52 53 54 55];
layer_3 = [14 15 16 21 24 29 33 38 41 46 47 48];
layer_4 = [22 23 30 32 39 40];
layer_5 = [31];


%% simualate the signal intensity obtained by all elements
% this part can be replaced with Kwave simulation.
int = zeros(1,61);
int(layer_1) = 0.6;
int(layer_2) = 0.7;
int(layer_3) = 0.8;
int(layer_4) = 0.9;
int(layer_5) = 1.0;

% Create a grid for the entire image
[grid_x, grid_y] = meshgrid(-400:1:400, -400:1:400);  % 100x100 image grid


%% interpolation algorithm
known_x = [x];
known_y = [y];
known_intensity = [int];

% Interpolate intensities on the grid
interpolated_intensity = griddata(known_x, known_y, known_intensity, grid_x, grid_y, 'cubic');

% Plot the interpolated image
figure;
axis_x = [-400:1:400];
axis_y = [-400:1:400];

imagesc(axis_x,axis_y,interpolated_intensity);
colormap hot;
colorbar;
title('Interpolated Intensity Image');
axis equal tight;

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