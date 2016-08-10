function featureVec = calculate_features(data,iFeatures,filters,extra)

% total energy
energy_total = sum(data(extra+1:end-extra,:).^2);
ind = 0;

featureVec = zeros(1,length(iFeatures));
for n = 1:5
    
    % check if we have features here
    check = ismember(iFeatures,16*(n-1)+1:16*n);
    if sum(check)
        % energy bands
        if n <= 4
            % work out which channels we need
            channels = iFeatures(check);
            channels = channels - 16*(n-1);
            % get the correct filter
            curFilter = filters(2*n-1:2*n,:);
            % filter the segment & cut it down
            temp = filtfilt(curFilter(1,:),curFilter(2,:),data(:,channels));
            temp = temp(extra+1:end-extra,:);
            % get the energy ratio
            features = sum(temp.^2) ./ energy_total(channels);
            featureVec(ind+1:ind+length(features)) = features;
            ind = ind + length(features);
            
            % line length
        else
            channels = iFeatures(check);
            channels = channels - 16*(n-1);
            features = sum(abs(diff(data(extra+1:end-extra,channels))));
            featureVec(ind+1:ind+length(features)) = features;
            ind = ind + length(features);
        end
        
    end % end feature check
    
end % end energy bands

end