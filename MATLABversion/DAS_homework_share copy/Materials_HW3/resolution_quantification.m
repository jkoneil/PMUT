clear all; close all; clc;

%% ---- Load and Beamform validation data ----
soundSpeed = 1540;
no_ele = 256;
channelSpacing = 100/no_ele;
fs = 40e6;
sampleSpacing = (1/fs)*soundSpeed*1e3;

% Load rf data from validation set
rfDat = [];
for i = 1:no_ele
    filename = fullfile(sprintf('rf_data_validation/rf_ln%01d.mat', i));
    load(filename);
    rfDat(1:size(rf_data,1),i) = rf_data;
end

% Run DAS beamforming
out = DAS(rfDat, no_ele, fs, channelSpacing, soundSpeed);

% Envelope detection
env = abs(hilbert(out));
env_norm = env ./ max(env(:));
env_db = db(env_norm);

%% ---- Setup axes ----
x_axis = (1:size(out,2)) * channelSpacing;  % lateral [mm]
y_axis = (1:size(out,1)) * sampleSpacing/2; % axial [mm]

%% ---- Find point targets ----
% Threshold to find bright spots (point targets)
threshold = -6; % dB
binary = env_db > threshold;

% Label connected regions
CC = bwconncomp(binary);
stats = regionprops(CC, env_db, 'Centroid', 'BoundingBox');

fprintf('Found %d point targets\n', length(stats));

%% ---- Quantify resolution for each point target ----
figure;
imagesc(x_axis, y_axis, env_db, [-60 0]);
colormap gray;
colorbar;
xlabel('Lateral [mm]');
ylabel('Axial [mm]');
title('Validation Phantom - Point Targets');
hold on;

% Store results
axial_res = zeros(1, length(stats));
lateral_res = zeros(1, length(stats));

for k = 1:length(stats)
    
    % Get centroid of point target
    cx = round(stats(k).Centroid(1)); % lateral index
    cy = round(stats(k).Centroid(2)); % axial index
    
    %% --- Axial Resolution ---
    axial_profile = env_db(:, cx);       % vertical slice through peak
    peak_val = max(axial_profile);
    threshold_val = peak_val - 6;        % -6dB point
    
    % Find indices above -6dB
    above = axial_profile >= threshold_val;
    indices = find(above);
    
    if ~isempty(indices)
        axial_fwhm_samples = indices(end) - indices(1) + 1;
        axial_res(k) = axial_fwhm_samples * sampleSpacing/2; % convert to mm
    end
    
    %% --- Lateral Resolution ---
    lateral_profile = env_db(cy, :);     % horizontal slice through peak
    peak_val_lat = max(lateral_profile);
    threshold_val_lat = peak_val_lat - 6; % -6dB point
    
    % Find indices above -6dB
    above_lat = lateral_profile >= threshold_val_lat;
    indices_lat = find(above_lat);
    
    if ~isempty(indices_lat)
        lateral_fwhm_samples = indices_lat(end) - indices_lat(1) + 1;
        lateral_res(k) = lateral_fwhm_samples * channelSpacing; % convert to mm
    end
    
    % Print results
    fprintf('\nPoint Target %d:\n', k);
    fprintf('  Location:          x = %.2f mm, y = %.2f mm\n', ...
             x_axis(cx), y_axis(cy));
    fprintf('  Axial Resolution:   %.4f mm\n', axial_res(k));
    fprintf('  Lateral Resolution: %.4f mm\n', lateral_res(k));
    
    % Mark on image
    plot(x_axis(cx), y_axis(cy), 'r+', 'MarkerSize', 10, 'LineWidth', 2);
    text(x_axis(cx)+0.5, y_axis(cy), sprintf('T%d', k), 'Color', 'red');
end

%% ---- Plot profiles of first point target ----
figure;

% Axial profile
subplot(1,2,1);
cx1 = round(stats(1).Centroid(1));
axial_profile1 = env_db(:, cx1);
plot(y_axis, axial_profile1, 'b-', 'LineWidth', 1.5);
yline(-6, 'r--', '-6dB threshold', 'LineWidth', 1.5);
xlabel('Axial [mm]');
ylabel('Amplitude [dB]');
title(sprintf('Axial Profile (Target 1)\nFWHM = %.4f mm', axial_res(1)));
grid on;
xlim([y_axis(round(stats(1).Centroid(2)))-2, ...
      y_axis(round(stats(1).Centroid(2)))+2]);

% Lateral profile
subplot(1,2,2);
cy1 = round(stats(1).Centroid(2));
lateral_profile1 = env_db(cy1, :);
plot(x_axis, lateral_profile1, 'b-', 'LineWidth', 1.5);
yline(-6, 'r--', '-6dB threshold', 'LineWidth', 1.5);
xlabel('Lateral [mm]');
ylabel('Amplitude [dB]');
title(sprintf('Lateral Profile (Target 1)\nFWHM = %.4f mm', lateral_res(1)));
grid on;

%% ---- Summary table ----
fprintf('\n========== RESOLUTION SUMMARY ==========\n');
fprintf('%-10s %-20s %-20s\n', 'Target', 'Axial Res (mm)', 'Lateral Res (mm)');
fprintf('----------------------------------------\n');
for k = 1:length(stats)
    fprintf('%-10d %-20.4f %-20.4f\n', k, axial_res(k), lateral_res(k));
end
fprintf('Mean Axial Resolution:   %.4f mm\n', mean(axial_res));
fprintf('Mean Lateral Resolution: %.4f mm\n', mean(lateral_res));