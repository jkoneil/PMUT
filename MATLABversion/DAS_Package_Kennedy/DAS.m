% DAS functions can be fine tuned.
% it easier to have aperture growth together with apodization

function out = DAS(rf, no_ele, Fs, channelSpacing, speedSound)

ns = size(rf,1); % number of samples
nl = size(rf,2); % number of lines

sampleSpacing = (1/Fs*speedSound)*1000; % for PA data

postBF = zeros(ns,nl);
RF = rf;

for n = 1:nl % lateral index of reconstructed pixel
    for m = 1:ns % axial index of reconstructed pixel
        for i = 1:no_ele % element number used for reconstruction
            if i > 0 && i < nl
                depth = m * sampleSpacing;
                width = abs(i - n) * channelSpacing;
                distance = sqrt(depth^2 + width^2);
                delay = distance/sampleSpacing;
                if delay < ns
                    y = RF(round(delay),i); % Go to sensor i, grab the signal corresponding to the pixel
                    if isnan(y)
                        y = 0;
                    end
                    postBF(m,n) = postBF(m,n) + y; % Add sensor's contribution to the pixel
                end
            end
        end
    end
end

out = postBF;