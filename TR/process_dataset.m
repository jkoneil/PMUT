% PROCESS_DATASET  One dataset, start to finish:
%   1) average frames by MUX   (PA_average_dataset)
%   2) map & merge to 61 ch    (PA_map_and_merge)
%   3) 3D reconstruction       (Python: TR/TR_Recon_Package_V1/TRHexArray3D_gpu.py, GPU if available)
%
% Click Run, pick the dataset folder (e.g. D:\Data_09252026\pmut_#46_hair_1), and wait.
% The dataset name comes from the folder name - nothing is hardcoded.
%
% Expected layout (this file, the two PA_*.m functions, Processed, TR and venv side by side):
%   SomeFolder\
%     process_dataset.m, PA_average_dataset.m, PA_map_and_merge.m
%     Processed\                      <- outputs of steps 1 and 2
%     TR\TR_Recon_Package_V1\         <- Python reconstruction
%     venv\                           <- Python environment

clear; close all; clc;

%% settings
num_avg     = 400;      % frames averaged per MUX
openSlicer  = true;     % show the result in 3D Slicer
pyOptions   = '';       % extra options for the reconstruction, e.g. '--zmax 8'

%% paths (works whether this file is in PMUT/ or in PMUT/TR/)
here   = fileparts(mfilename('fullpath'));
outDir = fullfile(here, 'Processed');

% reconstruction script: next to this file, or inside a TR folder next to it
reconScript = '';
for c = {fullfile(here, 'TR_Recon_Package_V1'), fullfile(here, 'TR', 'TR_Recon_Package_V1')}
    if isfile(fullfile(c{1}, 'TRHexArray3D_gpu.py'))
        reconScript = fullfile(c{1}, 'TRHexArray3D_gpu.py'); break
    end
end
assert(~isempty(reconScript), 'TR_Recon_Package_V1 not found next to %s', here);

% Python: a venv in this folder or up to two levels above; otherwise the system Python
if ispc, venvExe = fullfile('venv', 'Scripts', 'python.exe');
else,    venvExe = fullfile('venv', 'bin', 'python3'); end
pythonExe = '';
d = here;
for k = 1:3
    if isfile(fullfile(d, venvExe)), pythonExe = fullfile(d, venvExe); break; end
    d = fileparts(d);
end
if isempty(pythonExe) && ~ispc      % Mac without a venv: try the usual install locations
    for c = {'/Library/Frameworks/Python.framework/Versions/Current/bin/python3', ...
            '/opt/homebrew/bin/python3', '/usr/local/bin/python3'}
        if isfile(c{1}), pythonExe = c{1}; break; end
    end
end
assert(~isempty(pythonExe), 'Python not found - set up the venv (see README)');
fprintf('Using Python: %s\n', pythonExe);

%% pick the dataset folder
datasetDir = uigetdir(pwd, 'Pick the dataset folder (e.g. pmut_#46_hair_1)');
if isequal(datasetDir, 0), error('No folder selected'); end
[~, dataset_name] = fileparts(datasetDir);
fprintf('Dataset: %s\n', dataset_name);

%% 1) averaging
fprintf('1/3  Averaging (%d frames per MUX)...\n', num_avg);
tic; avgFile = PA_average_dataset(datasetDir, num_avg, outDir); toc

%% 2) map & merge
fprintf('2/3  Mapping and merging to 61 channels...\n');
tic; mergedFile = PA_map_and_merge(avgFile, outDir); toc

%% 3) reconstruction (Python)
fprintf('3/3  Reconstructing...\n');
cmd = sprintf('"%s" "%s" --data "%s" %s', pythonExe, reconScript, mergedFile, pyOptions);
if ~openSlicer, cmd = [cmd ' --no-slicer']; end
if ispc, cmd = ['"' cmd '"']; end
tic; [status, out] = system(cmd); toc
disp(out)
if status ~= 0
    error('Reconstruction failed (exit %d)', status);
end

fprintf('Done: %s\n', dataset_name);