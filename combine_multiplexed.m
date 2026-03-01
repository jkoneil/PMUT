% Combine 1024x16x4 multiplexed records into full data (1024x61)
function full_data = combine_multiplexed(multiplexed_data)
[num_depths, num_records, num_sections] = size(multiplexed_data); % Expected : [1024, 16, 4]

% Validate input size
if num_records ~= 16 || num_sections ~= 4
    error('Incorrect input size');
end

% Allocate output array
full_data = zeros(num_depths, 61);

%% Transfer data into a single, 2D output array
for rec = 1:16 % Loop over the 16 records
    for sec = 1:4 % Look at each section
        ch = (sec-1)*16 + rec; % Get channel index
        if ch <= 61 % Only 61 channels
            full_data(:, ch) = multiplexed_data(:, rec, sec); % Transfer all 1024 depth data
        end
    end
end

end