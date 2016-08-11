clear
close all
clc

iPt = 8;

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

% data_path = 'C:\Users\pkaroly\Dropbox\NV_MATLAB\LL-Prediction\TrainingData\';
data_path = 'TrainingData/';
save_path = [data_path Patient{iPt}];

% when training period ends
start_test = 4*4*7;   % (first four months is training period)


%% load information
curPt = Patient{iPt};
load([curPt '_DataInfo']);
trial_t0 = datenum(MasterTimeKey(1,:));
load(['Portal Annots/' curPt '_Annots']);

% chron. order
[SzTimes,I] = sort(SzTimes);
SzType = SzType(I);
SzDur = SzDur(I);

remove = SzType == 3;
SzType(remove) = [];
SzTimes(remove) = [];

% save seizures within training & test period
SzDay = ceil(SzTimes/1e6/60/60/24);
training = SzDay < start_test;

%% get the circadian time of seizures
SzCirc = trial_t0 + SzTimes/1e6/86400;
SzCirc = datevec(SzCirc);
SzCirc = SzCirc(:,4);

Seizures1 = SzCirc(training);
Seizures2 = SzCirc(~training);

% create the empirical histogram
SzHist = hist(Seizures1,0:23);
SzHist = SzHist + 1;            % need a uniform prior of 1 to avoid any zero probability times

% DOING THE WRAPPED HISTOGRAM
Kn = 24;         % number of kernels
Kmean = 0:23;    % kernel centers
Kbw = 0.6;         % this is the "bandwidth" or std of the Gaussian kernels
Kgauss = zeros(Kn,Kn);  % these are my distributions

for nn = 1:Kn
    Kgauss(nn,:) = SzHist(nn) * generate_circ_pdf(Kmean(nn),Kmean,1/Kbw);
end
% the density is given by the normalized sum of the kernels
SzProb = sum(Kgauss,2); % keep this as raw value so we can iterate the sum
SzProb = SzProb/trapz(0:23,SzProb);  % this is to normalize the area to 1 (ish)
save([save_path 'SzProb'],'SzProb','trial_t0')

%% Now pre-compute all the probablity updates
save_path = [data_path Patient{iPt} 'SzProbAll/'];
mkdir(save_path);
for iSz = 1:length(Seizures2)
    % create the empirical histogram
    SzHist = hist([Seizures1 ; Seizures2(1:iSz)],0:23);
    SzHist = SzHist + 1;            % need a uniform prior of 1 to avoid any zero probability times
    % params as above
    Kgauss = zeros(Kn,Kn);  % these are my distributions    
    for nn = 1:Kn
        Kgauss(nn,:) = SzHist(nn) * generate_circ_pdf(Kmean(nn),Kmean,1/Kbw);
    end
    % the density is given by the normalized sum of the kernels
    SzProb = sum(Kgauss,2); % keep this as raw value so we can iterate the sum
    SzProb = SzProb/trapz(0:23,SzProb);  % this is to normalize the area to 1 (ish)
    
 % save according to the number of seizures included in the pdf
    save([save_path 'SzProb' num2str(length(Seizures1) + iSz)],'SzProb')
end