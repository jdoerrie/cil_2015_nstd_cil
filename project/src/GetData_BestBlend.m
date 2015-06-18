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
[M, N] = size(X);

% Split intro training and testing index sets
idx = find(X ~= nil);
n = numel(idx);

rp = randperm(n);
rmses = zeros(nFolds, 1);
times = zeros(nFolds, 1);

[trnSplits, tstSplits] = CrossValidationSplits(rp, nFolds);

pool = parpool('local', 10);
% for K=1:20
% for K=[1,2,4,8,16,32,64,128,256]
fprintf('Best Blend\n');
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
  X_preds = zeros(M, N, 4);
  X_preds(:,:,1) = SVDpp(X_trn, 256, 0.002, 0.02, 1.0);
  X_preds(:,:,2) = FactNgbrItem(X_trn, 256, 0.01, 0.1, 0.900);
  X_preds(:,:,3) = FactNgbrUser(X_trn, 64, 0.01, 0.05, 0.975);
  X_preds(:,:,4) = 1;
  [X_pred, allW] = BinnedRidgeRegression(X_trn, X_preds, 1000, 100);
  times(k) = toc;

  % Compute RMSE
  rmses(k) = RMSE(X_pred, X_tst, nil);  % error on known test values
  fprintf('Fold %2d / %d: RMSE = %f, CPU = %f\n', ...
          k, nFolds, rmses(k), times(k));
  for i=1:4
    fprintf('Weight %d: %f +/- %f\n', i, mean(allW(i,:)), std(allW(i,:)));
  end
end

fprintf('RMSE: Mean = %f, Std = %f\n', mean(rmses), std(rmses));
fprintf('CPU:  Mean = %f, Std = %f\n', mean(times), std(times));

delete pool;
