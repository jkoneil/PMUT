function out = DAS(rf, no_ele, Fs, channelSpacing, speedSound)

ns = size(rf,1);   % number of samples
nl = size(rf,2);   % number of lines

pixelSpacing = (1/Fs*speedSound)*1000/2; % spacing between pixel

postBF = zeros(ns,nl);
for n = 1:nl % lateral number of reconstructed pixel
    RF = rf;
    for m = 1:ns % depth number of reconstructed pixel
        for i = 1:no_ele % element number used for the reconstruction
            if i > 0 && i <= nl
                depth = m * pixelSpacing;              % axial distance to pixel [mm]
                width = abs(i - n) * channelSpacing;   % lateral distance to pixel [mm]
                distance = sqrt(depth^2 + width^2);    % euclidean distance [mm]
                delay = distance/pixelSpacing;
                if delay < ns
                    y = RF(round(delay),i);
                    if isnan(y)
                        y=0;
                    end
                    postBF(m,n) = postBF(m,n) + y;
                end
            end
        end
    end
end

out = postBF;