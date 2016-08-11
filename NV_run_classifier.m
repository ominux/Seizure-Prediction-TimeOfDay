%% RUN CLASSIFIER on NEUROVISTA data

% function NV_run_classifier_VLSCI(iSeg,iPt)

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
iCh = 1:16;

featureWin = 5;       % time in seconds for feature vec calculation (SECONDS)
featureWinSlide = 1;  % sliding window amount for feature vec calculation (SECONDS)
extraTime = 0.5;        % for filtering artifact (SECONDS)
featureAvWin = 60;    % to average features (SECONDS)

% set filters for energy features
NV_filters;

%% set time info

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
% t0 = start_time*86400;
t0 = 115*86400;

% initialize
maxT = 1800;          % max amount from portal
out = zeros(2,segment_length);

N = segment_length / maxT;
iSec = 0;       % counts the seconds

for n = 1:N
    
    % shift timer depending on which chunk we're in (ref to absolute 0)
    t0 = t0 + maxT;
    
    % get the data from the portal
    try
        Data = getvalues(patient.data,t0 * 1e6,(maxT + 2*extraTime)* 1e6,iCh);
    catch
        % try again
        try
            Data = getvalues(patient.data,t0 * 1e6,(maxT + 2*extraTime) * 1e6,iCh);
        catch
            % maybe lost connection.
            display('early termination');
            continue;
        end
    end
    
    % pre-filter
    Data(isnan(Data(:,1)),:) = 0;
    Data = filtfilt(filter_wb(1,:),filter_wb(2,:),Data);
    
    % need to grab the data in segments
    for nn = 0:featureWinSlide:(maxT - featureWin)/featureWinSlide;
        
        ind1 = floor(Fs_actual*nn)+1;
        Win = floor(Fs_actual*(featureWin + 2*extraTime));
        curSeg = Data(ind1:ind1+Win,:);
        
        
        if sum(curSeg(:,1)) == 0
            % ignore dropout sections
            continue;
        end
        
        % calculate features from data (all 80 features)
        features = calculate_features(curSeg,iFeatures,filters,round(Fs_actual*extraTime));
        
    end % end feature segments
    
end