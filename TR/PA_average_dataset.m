function avgFile = PA_average_dataset(datasetDir, num_avg, outDir)
% PA_AVERAGE_DATASET  Combine Verasonics and NI time tags, group frames by MUX, and average.
% This is the quick-averaging script by Y. Tang (8-channel version) turned into a function;
% the algorithm is unchanged, only the hardcoded dataset name and paths were removed.
%
%   avgFile = PA_average_dataset(datasetDir, num_avg, outDir)
%
%   datasetDir : folder of ONE dataset, e.g. D:\Data_09252026\pmut_#46_hair_1
%                (contains <name>_timetag\, NI_timetag\ and <name>\ with the Frame_*.mat files)
%   num_avg    : number of frames to average per MUX (e.g. 400)
%   outDir     : folder to save <name>_<num_avg>_Averaged.mat in
%   avgFile    : full path of the saved file

[~, dataset_name] = fileparts(datasetDir);       % dataset name = folder name

folder_name_timetag_vera = fullfile(datasetDir, [dataset_name '_timetag']);
folder_name_timetag_ni   = fullfile(datasetDir, 'NI_timetag');
folder_name_data_vera    = fullfile(datasetDir, dataset_name);

%% load all time tags

% load tags from verasonics
files = dir(fullfile(folder_name_timetag_vera, '*.mat*'));   % all .mat files
files = files(~[files.isdir]);                                % exclude subdirectories
num_tag_vera = numel(files);

for i = 1:num_tag_vera
    S = load(fullfile(folder_name_timetag_vera, sprintf('Timetag_%d.mat', i)));
    time_tag_all_vera(i,:) = S.timeStamp; %#ok<AGROW>
end

% load tags from ni
num_tag_ni = 8;
for i = 1:num_tag_ni
    S = load(fullfile(folder_name_timetag_ni, sprintf('time_tag_%s_%d.mat', dataset_name, i)));
    time_tag_all_ni(i,:) = S.t_abs; %#ok<AGROW>
end

%% convert and sort
time_tag_all_vera_new = posixtime(time_tag_all_vera);
time_tag_all_ni_new   = posixtime(time_tag_all_ni);

table_1_vera = [zeros(size(time_tag_all_vera_new)), (1:num_tag_vera)', time_tag_all_vera_new];
table_1_ni   = [ones(size(time_tag_all_ni_new)),    (1:num_tag_ni)',   time_tag_all_ni_new];

table_1_comb = [table_1_vera; table_1_ni];
table_1_sort = sortrows(table_1_comb, 3);

%% compile table_2 for easy averaging
divider_idx = find(table_1_sort(:,1) == 1);
boundaries  = [0; divider_idx; size(table_1_sort,1)];

table_2 = {};
for i = 1:length(boundaries)-1
    row_start = boundaries(i) + 1;
    row_end   = boundaries(i+1) - 1;
    if row_start <= row_end
        table_2{end+1} = table_1_sort(row_start:row_end, 2); %#ok<AGROW>
    end
end

%% averaging (same averaging number for all MUXs)
for i = 1:(size(table_2,2)-1)
    list_data_idx = table_2{i};

    start_idx = size(list_data_idx,1) - num_avg + 1;
    end_idx   = size(list_data_idx,1);
    if start_idx < 1                                  % sanity check
        start_idx = 1;
    end
    num_avg_real = end_idx - start_idx + 1;

    for j = start_idx:end_idx
        data_idx = list_data_idx(j);
        S = load(fullfile(folder_name_data_vera, sprintf('Frame_%d.mat', data_idx)));
        data_now = double(S.RFdata(:, [1, 61, 68, 128, 129, 189, 196, 256]));   % double avoids truncation

        if j == start_idx
            data_all = data_now;
        else
            data_all = data_all + data_now;
        end
    end

    data_avg(:,:,i) = data_all / num_avg_real; %#ok<AGROW>
end

%% data saving
if ~isfolder(outDir), mkdir(outDir); end
avgFile = fullfile(outDir, sprintf('%s_%d_Averaged.mat', dataset_name, num_avg));
save(avgFile, 'data_avg', '-v7');
fprintf('  saved %s\n', avgFile);
end