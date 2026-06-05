clear all;

soundSpeed = 1540;
no_ele = 256; % number of elements
channelSpacing = 100/no_ele; % image width: 100mm
fs = 40e6; %sample frequency
sampleSpacing = (1/fs)*soundSpeed*1e3; %sample number vs mm

for i = 1:no_ele
    filename = fullfile(sprintf('rf_data_256/rf_ln%01d.mat', i));
    load(filename);
    rfDat(1:size(rf_data,1),i) = rf_data;
end

%% Calculate intensity data in each pixel with Delay-and-Sum
out = DMAS(rfDat, no_ele, fs, channelSpacing, soundSpeed);

%% Convert from intensity data to decibel data and show B-mode image
env = abs(hilbert(out));
st = 100;
ed = 5200;
x = [1 size(out,2)]*channelSpacing;
y = [st ed]*sampleSpacing/2;
env = env/max(max(env(st:ed,:)));
figure
imagesc(x,y,db(env(st:ed,:)),[-60 0]);
colormap(gray)
axis image
colorbarseriously 