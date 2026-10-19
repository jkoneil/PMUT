function out = DMAS(rf, no_ele, Fs, channelSpacing, speedSound)
% Delay-Multiply-and-Sum (F-DMAS) Beamforming
% Based on: Matrone et al., IEEE TMI, 2015
% Implements equations (2)-(4) from the paper

ns = size(rf,1);   % number of samples
nl = size(rf,2);   % number of lines
pixelSpacing = (1/Fs*speedSound)*1000/2; % [mm] spacing between pixels
postBF = zeros(ns, nl);

for n = 1:nl          % lateral pixel index
    RF = rf;
    
    % Step 1: Collect delayed signals for ALL elements (same as DAS)
    delayed = zeros(ns, no_ele);
    
    for i = 1:no_ele
        if i > 0 && i <= nl
            for m = 1:ns
                depth    = m * pixelSpacing;
                width    = abs(i - n) * channelSpacing;
                distance = sqrt(depth^2 + width^2);
                delay    = distance / pixelSpacing;
                
                if delay < ns
                    y = RF(round(delay), i);
                    if isnan(y)
                        y = 0;
                    end
                    delayed(m, i) = y;
                end
            end
        end
    end
    
    % Step 2: DMAS - combinatorial pair multiplication
    % Equation (2): all unique pairs (i,j) where j > i
    % Equation (3): signed square root to restore signal dimensionality
    % Equation (4): sum all pair contributions
    
    dmas_line = zeros(ns, 1);
    
    for i = 1:no_ele-1
        for j = i+1:no_ele
            % Eq. (3): signed square root of product
            product = delayed(:,i) .* delayed(:,j);
            s_hat = sign(product) .* sqrt(abs(product));
            
            % Eq. (4): accumulate
            dmas_line = dmas_line + s_hat;
        end
    end
    
    postBF(:, n) = dmas_line;
end

% Step 3: Bandpass filter around 2*f0 (F-DMAS)
% Multiplication doubles the center frequency from f0 to 2*f0
f0 = 2e6;           % transducer center frequency [Hz]
f_center = 2 * f0;  % DMAS output centered at 2*f0 = 4 MHz
bw = 0.6;           % fractional bandwidth

f_low  = f_center * (1 - bw/2);  % lower cutoff
f_high = f_center * (1 + bw/2);  % upper cutoff

% Design bandpass filter
[b, a] = butter(4, [f_low, f_high]/(Fs/2), 'bandpass');

% Apply filter to each scan line
for n = 1:nl
    postBF(:,n) = filtfilt(b, a, postBF(:,n));
end

out = postBF;