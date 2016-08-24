%% CLASSIFIER TRAINING on NEUROVISTA data
%
%
clear
close all
clc
%% ALGORITHM PARAMETERS

% data
Nfeatures = 16;         % number of final features to choose
averageSize = 60;       % time in seconds to take average over

% cross validation
Nfolds = 10;                 % number of cross validations (at the moment using randperm)
test_percent = 1/Nfolds;     % size of test percent

% log regression
lambda = 0.5;   % regularization param
MaxIter = 100; % grad descent iterations
featureNormalize = 1;  % pre normalize the features? zero mean unit var

%
% this is the test patient
patient = '24_002';
% load data
load(['TrainingData/' patient 'TrainingSeizures']);
load(['TrainingData/' patient 'TrainingNonSeizures']);

NSz = size(preIctal,3);
T = size(preIctal,2);
Slide = averageSize/2;
N = 2 * (T-Slide)/averageSize;

% average in chunks
all_data_train = zeros(80,N,2*NSz);
all_data_labels = zeros(N,2*NSz);
for n = 1:N
    ind1 = Slide*(n-1)+1;
    % INTERICTAL
    temp = interIctal(:,ind1:ind1+averageSize-1,:);
    dropouts = interIctalDropouts(ind1:ind1+averageSize-1,:);
    dropouts = averageSize - sum(dropouts == 1);
    if sum(dropouts==0)
        display('update code')
        return;
    end
    % take the weighted average by removing the dropouts
    temp = squeeze(sum(temp,2));
    temp = temp ./ repmat(dropouts,80,1);
    all_data_train(:,n,1:NSz) = temp;
    all_data_labels(n,1:NSz) = 0;  %% 0 for non-seizure
    % PRE-ICTAL
    temp = preIctal(:,ind1:ind1+averageSize-1,:);
    dropouts = preIctalDropouts(ind1:ind1+averageSize-1,:);
    dropouts = averageSize - sum(dropouts == 1);
    % take the weighted average by removing the dropouts
    temp = squeeze(sum(temp,2));
    temp = temp ./ repmat(dropouts,80,1);
    all_data_train(:,n,NSz+1:end) = temp;
    all_data_labels(n,NSz+1:end) = 1;  % 1 for seiuzre
end

% reshape the data
all_data_train = reshape(all_data_train,80,N*2*NSz);
all_data_labels = reshape(all_data_labels,1,N*2*NSz);


%% Independence criteria
% narrow feature vector down to 16 features
if featureNormalize
    mu = mean(all_data_train,2);
    sigma = std(all_data_train,[],2);
    for n = 1:80
        all_data_train(n,:) = (all_data_train(n,:) - mu(n)) ./ sigma(n);
    end
    
end

corrX = mean(abs(corr(all_data_train')));
[~,I] = sort(corrX);  % ascending
iFeatures = I(1:Nfeatures);  % takes the Nfeatures smallest correlations
% get it back into chron. order
iFeatures = sort(iFeatures);

% get the mean & std
mu = mu(iFeatures);
sigma = sigma(iFeatures);

%% Train Logistic Regression
all_data_train = all_data_train(iFeatures,:);
[W_base,~,~] = logistic_regression_fit(all_data_train',[],all_data_labels,[],MaxIter,lambda,0);


%% Shift the Log Regression based on time of day
load(['TrainingData/' patient 'SzProb']);

% we are only going to shift the intercept - i.e. the first weight
W_weighted = repmat(W_base,1,24);
% the observed probability of seizure based on the training data
Pobs = sum(all_data_labels) / length(all_data_labels);
for n = 1:24
    % the probability of seizure at this time
    Ps = SzProb(n);
    W_weighted(1,n) = W_weighted(1,n) - log(Pobs/(1-Pobs) * (1-Ps)/Ps);
end


%% save the results
save(['TrainingData/' patient 'Classifier'],'W_base','W_weighted','featureNormalize','SzProb', ...
    'iFeatures','Pobs','Nfeatures','mu','sigma');
