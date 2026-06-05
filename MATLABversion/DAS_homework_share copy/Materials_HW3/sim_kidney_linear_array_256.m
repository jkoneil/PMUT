
%  Phased array B-mode scan of a human kidney
%
%  This script assumes that the field_init procedure has been called
%  Here the field simulation is performed and the data is stored
%  in rf-files; one for each rf-line done. The data must then
%  subsequently be processed to yield the image. The data for the
%  scatteres are read from the file pht_data.mat, so that the procedure
%  can be started again or run for a number of workstations.
%
%  Generate the transducer apertures for send and receive
% 
%  Noted that need to generate phantom data (pht_data.mat) by runnnig
%  make_scatterers.m before running this procedure

clear all
close all
addpath(genpath('Field_II_mac'))                % Load libraries for Mac
% addpath(genpath('Field_II_windows'))          % Load libraries for Windows

% speed sound
p.c=1540;                  %  Speed of sound [m/s]
p.fs=40e6;                %  Sampling frequency [Hz]
p.datDecimate = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% TRANSDUCER PARAMS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%
p.f0 = 2e6;                  %  Transducer center frequency [Hz]
% p.fractionalBW = 0.6
p.fractionalBW = 0.6
p.txCycles = 1.0;
p.lambda = p.c/p.f0;

p.focusTx=[0 0 60]/1000;     %  Fixed focal point [m]

p.pitch_x=p.lambda/2;
p.kerf_x=p.lambda/100;          %  Kerf [m]
p.pitch_y = 0.8e-2; %p.lambda/2; %5e-3;

p.no_elements_x_rx = 1;
p.no_elements_x_tx = 1;
p.no_elements_y_rx = 1;
p.no_elements_y_tx = 1;

p.width = p.pitch_x - p.kerf_x; %lateral element width
p.height = p.pitch_y; % - p.kerf_y; %elevational element width
p.no_sub_x = 1;
p.no_sub_y = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INITITIALIZE AND CONFIGURE FIELD %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
field_init(-1);

set_field('c', p.c);           %set speed of sound in tissue
set_field('fs', p.fs);         %set sampling frequency

%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE APERTURES %%%
%%%%%%%%%%%%%%%%%%%%%%
% 
tx = xdc_linear_array(p.no_elements_x_tx, p.width, p.height, p.kerf_x, p.no_sub_x, p.no_sub_y, p.focusTx);
rx =  xdc_linear_array(p.no_elements_x_rx, p.width, p.height, p.kerf_x, p.no_sub_x, p.no_sub_y, p.focusTx);

%%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE XDCR PULSE %%%
%%%%%%%%%%%%%%%%%%%%%%%

tc = gauspuls('cutoff', p.f0, p.fractionalBW, -6, -60);
t = -tc:1/p.fs:tc;
ncycles = 1./p.fractionalBW;
omega = 2*pi*p.f0;
[ign, excitationPulse] = gauspuls(t, p.f0, p.fractionalBW);
p.excitationPulse = excitationPulse;
impulseResponse = 1;

p.tshift = ceil(size(conv(impulseResponse, conv(impulseResponse, ...
        excitationPulse)),2)/2)/p.fs; % # of samples to be offset of 2 convs (N+M-1) + M -1 = N + 2*M - 2

xdc_impulse(tx,impulseResponse);
xdc_impulse(rx,impulseResponse);
xdc_excitation(tx,excitationPulse);


%%   Load the computer phantom
load('pht_data.mat')

p.pos = phantom_positions;
p.amp = phantom_amplitudes;

positions = p.pos;
amp = p.amp;
scats = length(amp)

%%   Set the number and pitch of transducer elements
image_width = 0.1; % phantom width [m]
no_element = 512; % number of transducer element

p.delta_x = image_width./(no_element-1)

delta_x(1:scats, 1) = p.delta_x;
delta_y(1:scats, 1) = zeros;
delta_z(1:scats, 1) = zeros;
delta = [delta_x, delta_y, delta_z];


%% Generate RF data corresponding to each transducer element

for ii=1:no_element

 posII = positions(:,:) + (no_element-1)/2*delta - delta*(ii-1);

if ~exist(['rf_data_512/rf_ln',num2str(ii),'.mat'])

dirname = 'rf_data_512/';
mkdir(dirname)
else
    dirname = 'rf_data_512/';
end

[rf,tstart] = calc_scat(tx,rx,posII,amp);

propDiffSamples = (tstart-p.tshift)*p.fs;
propIntSamples = round(propDiffSamples);          %Gross shift

rf_data = padarray(rf, [propIntSamples, 0], 'pre');

%  Store the result
filename = fullfile(dirname, ['rf_ln', sprintf(num2str(ii),  ii),'.mat'])
save(fullfile(filename), 'rf_data', 'tstart', 'p', '-mat')
 
end