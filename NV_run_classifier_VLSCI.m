%% RUN CLASSIFIER on NEUROVISTA data

function NV_run_classifier_VLSCI(iSeg,iPt)

iPt = 8;

tic

addpath('IEEGToolbox');
addpath('IEEGToolbox/lib');
addpath('ieeg-cli-1.13');
addpath('ieeg-cli-1.13/config');
addpath('ieeg-cli-1.13/lib');

%% RUN CLASSIFIER on NEUROVISTA data

% Patients
Patient{1} = '23_002';
Patient{2} = '23_003';
Patient{3} = '23_004';
Patient{4} = '23_005';
Patient{5} = '23_006';
Patient{6} = '23_007';

Patient{7} = '24_001';
Patient{8} = '24_002';
Patient{9} = '24_004';
Patient{10} = '24_005';

Patient{11} = '25_001';
Patient{12} = '25_002';
Patient{13} = '25_003';
Patient{14} = '25_004';
Patient{15} = '25_005';

curPt = Patient{iPt};
patient = IEEGSession(['NVC1001_' curPt '_2'],'pkaroly','pkiEEG.bin');

% load pt information
% load('portal_index.mat');  % time_matrix, segment_length
load(['TrainingData/' curPt 'Classifier']);
load(['TrainingData/' curPt 'SzProb']);
load(['Portal Annots/' curPt '_Annots']);
SzTimes = SzTimes / 1e6;    % get to seconds
Pobs = repmat(Pobs,1,24);   % this is the probability of sz from teh training data

%% algorithm parameters
start_time = 4*4*7;   % (first four months is training period)

Fs_actual = patient.data.sampleRate;
Fs = 400;  % for filtering

featureWin = 5;       % time in seconds for feature vec calculation (SECONDS)
featureWinSlide = 1;  % sliding window amount for feature vec calculation (SECONDS)
extraTime = 1;        % for filtering artifact (SECONDS)
featureAvWin = 60;    % to average features (SECONDS)

% set filters for energy features
NV_filters;

%% set time info
t_test = start_time*86400; % where the testing phase starts from (day -> s)

% time of day
start_date = datevec(trial_t0);
cur_hour = start_date(4);
Nhours = start_date(5)/60 + start_date(6)/3600;  % number of hours passed since start time

% NB our sz probability vector is SzProb = [00:00 01:00 02:00 ... 23:00].
% so need to add one to the indexing

% portal_times = time_matrix{iPt};  % this is all job start times (s) for the patient
% t_abs_0 = portal_times(iSeg);     % this is for the current job - T absolute zero THIS VALUE SHOULD NOT BE EDITED
% t_abs_0 = t_test + t_abs_0 - featureAvWin;      % adjust forward by training period & back to give some overlap for averaging
% segment_length = 24*60*60*segment_length; % get to seconds (from days)


%% start grabbing data

segment_length = 5*60*60; % (s)

% initialize
maxT = 1800;          % max amount from portal
out = zeros(2,segment_length);

N = segment_length / maxT;
iSec = 0;       % counts the seconds

for n = 1:ceil(segment_length/maxT);
    
    % update the circadian timer
    cur_hour = mod(cur_hour + maxT/60/60,24);
    
    % shift timer depending on which chunk we're in (ref to absolute 0)
    t0 = t_abs_0 + maxT*(n-1) - 2*extraTime;
    
    % get the data from the portal
    try
        curData = getvalues(patient.data,t0 * 1e6,maxT * 1e6,1:16);
    catch
        % try again
        try
            curData = getvalues(patient.data,t0 * 1e6,maxT * 1e6,1:16);
        catch
            % maybe lost connection. save what you can
            display('early termination');
            out = out(:,1:iSec);
            save([curPt '_output'],'out','iSec');
            return;
        end
    end
       
    % set dropouts to zero
    dropouts = find(isnan(curData(:,1)));
    curData(dropouts,:) = 0;
    
    % filter data - Wideband & notch filter first
    curData = filtfilt(filter_wb(1,:),filter_wb(2,:),curData);
    curData = filtfilt(filter_notch(1,:),filter_notch(2,:),curData);
    
    % extra filter
    extraSamples = Fs * extraTime;
    featureSamples = Fs * featureWin;
    N = length(curData) - featureSamples - extraSamples;
    
    % number of seizures each block
    NSz = 0;
    tshift = 0;
    
    % start caluclating the features
    % keep a running average
    if n == 1
        fMeanVec = zeros(featureAvWin,Nfeatures);
        count = 0;
    end
    
    % loop through and calculate features
    featureLoop = extraSamples:Fs*featureWinSlide:N;
    for nn = featureLoop
               
        % use the real Fs here for timing
        tref = t0 + (nn+1)/Fs_actual;
        
        if n == 1 && nn == featureLoop(1)
            % the actual real start time
            t_0 = tref;
        elseif n == ceil(segment_length/maxT) && nn == featureLoop(end)
            % the actual real end time
            t_f = t0 + (nn+featureSamples)/Fs_actual;
        end
        
        data_ind = nn+1-extraSamples:nn+featureSamples+extraSamples;
        % skip if there are dropouts in the segment
        if sum(ismember(data_ind,dropouts));
            continue;
        end

        % get our 5 second window & calculate features
        curSeg = curData(data_ind,:);
        feature_vec = calculate_features(curSeg,iFeatures,filters,extraSamples);
        
        % the running sum to take an average
        if count < 60
            count = count + 1;
            fMeanVec(count,:) = feature_vec;
        elseif count == 60
            % the first mean
            fMean = mean(fMeanVec);
            count = 61;
        else
            % update the running mean iteratively
            fMean = fMean + (feature_vec - fMeanVec(1,:))/featureAvWin;
            fMeanVec = [fMeanVec(2:end,:) ; feature_vec];
        end
        
        % classify our data
        if count > 60
            iSec = iSec + 1;
            % check if theres a seizure in this second
            sz = sum(SzTimes > floor(tref) & SzTimes < (floor(tref) + 2));
            if sz > 0
                if (tref - tshift) > 2
                    NSz = NSz + 1;
                end
                tshift = tref;
                out(1,iSec) = -1;
                out(2,iSec) = -1;
            else
                out(1,iSec) = logistic_regression_run(W_base,fMean,1);
                out(2,iSec) = logistic_regression_run(W_weighted(:,floor(cur_hour)+1),fMean,1);
            end
        end
            
    end
    
    % check if there's been a seizure and update the distribution
    if NSz > 0
        SzProb = SzProb + NSz * time_pdf(:,floor(cur_hour)+1);
        % renormalize
        SzProb = SzProb ./ trapz(0:23,SzProb);
        % we are only going to shift the intercept - i.e. the first weight
        W_weighted(1,:) = repmat(W_base(1),1,24) - log(Pobs ./(1-Pobs) .* (1-SzProb')./SzProb');
    end    
    
end % end iEEG segments ( 2000 seconds )

%% save
save([curPt '_output_' num2str(iSeg)],'out','t_0','t_f');
toc

end