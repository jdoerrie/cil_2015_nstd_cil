% Evaluation script for the collaborative filtering problem. Loads data
% matrix, splits known values into training and testing sets, and computes
% the MSE of the predicted to the true known entries of the test data.
%
% Loads data from Data.mat and calls PredictMissingValues.m.

% Setup
rand('seed', 1);  % fix random seed for reproducibility

% Constants
filename = '../data/Data.mat';

% number of folds usedd for cross-validation
nFolds = 10;

% global nil;
nil = 0;  % missing value indicator

% Load data
L = load(filename);
X = L.X;

% Split intro training and testing index sets
idx = find(X ~= nil);
n = numel(idx);

rp = randperm(n);
rmses = zeros(nFolds, 1);
times = zeros(nFolds, 1);

[trnSplits, tstSplits] = CrossValidationSplits(rp, nFolds);

pool = parpool('local', 10);
% for K=1:20
fprintf('FactNgbrItem Data Collection\n');
fprintf('K,Mean_RMSE,Std_RMSE,Mean_CPU,Std_CPU\n');

for K=[1,2,4,8,16,32,64,128,256]
  parfor k=1:nFolds
    idx_trn = idx(trnSplits(k,:));
    idx_tst = idx(tstSplits(k,:));

    % Build training and testing matrices
    X_trn = ones(size(X))*nil;
    X_trn(idx_trn) = X(idx_trn);  % add known training values

    % global X_tst;
    X_tst = ones(size(X))*nil;
    X_tst(idx_tst) = X(idx_tst);  % add known training values

    % Predict the missing values here!
    tic;
    nils = X_trn == nil;
    X_trn(nils) = NaN;
    X_pred = FactNgbrItem(X_trn, K);
    times(k) = toc;

    % Compute RMSE
    rmses(k) = RMSE(X_pred, X_tst, nil);  % error on known test values
    fprintf('Fold %2d / %d: RMSE = %f, CPU = %f\n', ...
            k, nFolds, rmses(k), times(k));
  end

  fprintf('%d,%f,%f,%f,%f\n', K, mean(rmses), std(rmses), mean(times), std(times));
end

delete(pool);
