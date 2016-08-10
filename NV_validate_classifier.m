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
    
    for n = 1:80
        all_data_train(n,:) = (all_data_train(n,:) - mean(all_data_train(n,:))) ./ ...
            std(all_data_train(n,:));
    end
    
end

corrX = mean(abs(corr(all_data_train')));
[~,I] = sort(corrX);  % ascending
iFeatures = I(1:Nfeatures);  % takes the Nfeatures smallest correlations
% get it back into chron. order
iFeatures = sort(iFeatures);

%% Train Logistic Regression
all_data_train = all_data_train(iFeatures,:);
N = length(all_data_train);
Ntest = round(test_percent*N);

% intialize AUC values for each fold
AUC = zeros(1,Nfolds);
test_ind = randperm(N);
% 10 fold cross validation (w/o replacement)
for n = 1:Nfolds
    
    % index for test set
    if n == Nfolds
        this_fold = test_ind(Ntest*(n-1)+1:end);
    else
        this_fold = test_ind(Ntest*(n-1)+1:Ntest*n);
    end
    % test set
    data_test = all_data_train(:,this_fold);
    data_test_labels = all_data_labels(this_fold);
    % intialize training data to everything
    data_train = all_data_train;
    data_train(:,this_fold) = [];
    data_train_labels = all_data_labels;
    data_train_labels(this_fold) = [];
    
    [W,out,AUC(n)] = logistic_regression_fit(data_train',data_test',data_train_labels,data_test_labels,MaxIter,lambda,0);
    
    fprintf('cross validation fold %d ... \n',n)    
end

fprintf('\n average AUC is: %.3f \n',mean(AUC))