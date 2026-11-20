function mergedFile = PA_map_and_merge(avgFile, outDir)
% PA_MAP_AND_MERGE  Map MUX-channel data onto the 61 pMUT elements, filter, and interpolate.
% This is Map_and_Merge_V6 turned into a function; the algorithm is unchanged, only the
% hardcoded dataset name and paths were removed.
%
%   mergedFile = PA_map_and_merge(avgFile, outDir)
%
%   avgFile    : the <name>_<N>_Averaged.mat file from PA_average_dataset
%                (data_avg: samples x MUXs(8) x channels(8))
%   outDir     : folder to save <name>_PMUT_61_channels_64.mat in
%   mergedFile : full path of the saved file
%
% Output variables: data_raw (unfiltered), data_filt (11-16 MHz band-pass),
% data_raw_interp, data_filt_interp, upsampleFactor
% Uses the mapping from the new batch of pMUT dies.

[~, base] = fileparts(avgFile);
dataset_name = regexprep(base, '_\d+_Averaged$', '');    % strip "_<N>_Averaged"

S = load(avgFile);
data_avg = S.data_avg;

%% mapping from MUX to PMUT (new batch of pMUT devices)
lut_table = zeros(8,8);
lut_table(1:8,1) = [58, 53, 47, 40, 57, 51, 52, 44]';
lut_table(1:7,2) = [45, 46, 36, 37, 38, 39, 27]';
lut_table(1:8,3) = [19, 28, 12, 20, 29, 6, 13, 21]';
lut_table(1:8,4) = [30, 1, 2, 7, 3, 8, 14, 31]';
lut_table(1:8,5) = [4, 9, 15, 22, 5, 11, 10, 18]';
lut_table(1:7,6) = [17, 16, 26, 25, 24, 23, 35]';
lut_table(1:8,7) = [43, 34, 50, 42, 33, 56, 49, 41]';
lut_table(1:7,8) = [32, 61, 60, 55, 59, 54, 48]';

%% map into pMUT index (raw data, no filtering)
data_raw = zeros(size(data_avg,1), 61);
for i = 1:8          % MUX
    for j = 1:8      % channel
        pmut_channel = lut_table(j,i);
        if pmut_channel ~= 0
            data_raw(:,pmut_channel) = data_avg(:,i,j);
        end
    end
end

%% bandpass filter
fs = 62.5e6;
f_low = 11e6;
f_high = 16e6;
order = 8;

Wn = [f_low, f_high] / (fs/2);
[z, p, k] = butter(order, Wn, 'bandpass');
sos = zp2sos(z, p, k);

data_filt = zeros(size(data_raw));
for col = 1:size(data_raw,2)
    data_filt(:, col) = filtfilt(sos, 1, data_raw(:,col));
end

%% interpolation
upsampleFactor = 10;
data_raw_interp  = interpft(data_raw,  size(data_raw,1)  * upsampleFactor, 1);
data_filt_interp = interpft(data_filt, size(data_filt,1) * upsampleFactor, 1);

%% save
if ~isfolder(outDir), mkdir(outDir); end
mergedFile = fullfile(outDir, [dataset_name '_PMUT_61_channels_64.mat']);
save(mergedFile, 'data_raw', 'data_filt', 'data_raw_interp', 'data_filt_interp', 'upsampleFactor', '-v7');
fprintf('  saved %s\n', mergedFile);
end