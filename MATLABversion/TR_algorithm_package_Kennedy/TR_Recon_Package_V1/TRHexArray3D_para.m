clear; clc;

%inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data=importdata('Processed/d3_PMUT_61_channels_400.mat');%import the PA signal dataset
PASignal=Data.data_all(491:900,:);%use the start of PA signal generated in pMUT by the laser pulse as inernal trigger (set the "zero" time). 

% PASignal(:,10)=0;%remove noisey channels (for n5 only)
PASignal(:,52)=0;
PASignal(:,34)=0;

% Limit the usable signal length to slightly over 500 sample counts (depth of 12.5 mm at 50 MHz sampling rate)
PASignal(1:79,:)=0;%remove the pMUT reverb from laser pulse

%%
coord=importdata('coordsHexagonal0p5mm.txt');%import the coordinates (X,Y) of the pMUT element centers
co=1.5;%speed of sound, mm/us; 1.5 in water at room temperature; adjust depending on water temperature; 1.52-1.56 in tissue
dt=0.02;%time step, microseconds; corresponds to 50 MHz sampling rate. 
dxo=0.05;%3D reconstruction voxel size in x direction, mm; in simulations this was 0.01, or 10 microns, with 5-nanosecond time step (200 MHz sampling rate equivalent)
dyo=0.05;%3D reconstruction voexl size in y direction, mm
dzo=0.05;%3D reconstruction voxel size in z direction, mm
zmin=0;%axial space bounds, mm 
zmax=13;
xmax=7;%lateral space bound, mm 
ymax=7;%lateral space bound, mm 
xo=-xmax:dxo:xmax;
yo=-ymax:dyo:ymax;
zo=zmin:dzo:zmax;

Img=zeros(length(zo),length(xo),length(yo));%matrix placeholder for the brightness values in the 3D PA image

ThetaMax=pi/6;%maximum half-angle of the reconstructeed image (60 degrees sector assumed for now)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%time reversal reconstruction process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parpool('local',8); % initialize parallel computing


parfor XY=1:length(coord)%going through pMUT detectors, one by one; parallelize this step for faster reconstruction
    XY
    Xsens=coord(XY,1)*0.001;%pMUT coordinates in mm
    Ysens=coord(XY,2)*0.001;
    Receive=diff(PASignal(:,XY));%take a derivative of detected PA signal; NOTE: if the detected signal is NOT inverted, add a (-) 
    NumPoints=zeros(length(zo),length(xo),length(yo)); %temporary matrix for the number of signal samples fitting into a voxel
    ImgTemp=zeros(length(zo),length(xo),length(yo));%temporary matrix for the PA signal contribution from the selected pMUT

    for t=1:length(Receive) %number of PA signal sample
        l=(double(t))*dt*co; %radius of the arch corresponding to backpropagation
        dTheta=pi/(l*pi/dxo);%angular step in reconstruction - depends on the distance; becomes smaller with distance
        dphi=dTheta;%polar angle step in reconstruction
        for phi=0:dphi:(2*pi-dphi)
            for Theta=-ThetaMax:dTheta:ThetaMax 
                if ((l*cos(Theta)-zmin)>=0)
                    xind=int32((l*sin(Theta)*cos(phi)+Xsens+xmax)/dxo+1);
                    yind=int32((l*sin(Theta)*sin(phi)+Ysens+ymax)/dyo+1);
                    zind=int32((l*cos(Theta)-zmin)/dzo+1);
                    %if (Receive(t)>0)%alternative option - if only the
                    %positive parts of the PA signal are used in the
                    %reconstruction. Leads to worse lateral resolution.
                        ImgTemp(zind,xind,yind)=ImgTemp(zind,xind,yind)+l*Receive(t);
                        NumPoints(zind,xind,yind)=NumPoints(zind,xind,yind)+1;
                    %end
                end
            end
        end
    end

    Img=Img+ImgTemp./(NumPoints+1);
end
toc

%image display (suggest adding Matlab's 3D Viewer App)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

delete(gcp('nocreate'));

%visualize a 2D slice
figure
%imagesc(xo,zo,Img(:,:,117)) %colormap-based image of a 2D slice corresponding to plane Y=0

%%
imagesc(yo,zo,squeeze(Img(:,132,:)))%colormap-based image of a 2D slice corresponding to plane X=0
Y0Max=max(max(Img(:,:,117)));%Max brightness in 2D slice Y=0
X0Max=max(max(Img(:,132,:)));%Max brightness in 2D slice X=0
ImgMax=max(max(max(Img)));%global 3D image brightness max
clim([0 X0Max])%do not display the negative brightness values (artifactual)
clim([0.5*X0Max X0Max])%optional: only display within -6dB of brightness
%clim([0.5*Y0Max Y0Max])
ax = gca;
ax.YDir = 'normal';
axis image
colormap 'hot'
xlabel('y: [mm]');
ylabel('z: [mm]');
colorbar;

%visualize a 3D isosurface at -6dB (or half) brightness level
figure
Img1=permute(Img,[2,3,1]);
[X,Y,Z]=meshgrid(xo,yo,zo);
isosurface(X,Y,Z,Img1,ImgMax/2)
hold on
scatter(coord(:,1)*0.001,coord(:,2)*0.001)


%% Plot the 3D interpolated images
% figure;
% h = volshow(Img1, 'Colormap', hot);
% h.Alphamap = linspace(0,0.1,256)'.^1.3;  % min opacityf  = 0.
