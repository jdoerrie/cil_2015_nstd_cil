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

nil = 0;  % missing value indicator

% Load data
L = load(filename);
X = L.X;

nils = X == nil;
X(nils) = NaN;
idx = find(~nils);
n = numel(idx);

rp = randperm(n);
rmses = zeros(nFolds, 1);
times = zeros(nFolds, 1);

% Split intro training and testing index sets
[trnSplits, tstSplits] = CrossValidationSplits(rp, nFolds);

idxs_trn = zeros(size(trnSplits));
idxs_tst = zeros(size(tstSplits));

for k=1:nFolds
  idxs_trn(k,:) = idx(trnSplits(k,:));
  idxs_tst(k,:) = idx(tstSplits(k,:));
end

pool = parpool('local', 2);

fprintf('Grid Search Compute Biases\n');
fprintf('Lambda_1,Lambda_2,Mean_RMSE,Std_RMSE,Mean_CPU,Std_CPU\n');
for lam1=0:20
  for lam2=0:20
    parfor k=1:nFolds
      idx_trn = idxs_trn(k,:);
      idx_tst = idxs_tst(k,:);

      % Build training and testing matrices
      X_trn = ones(size(X))*NaN;
      X_trn(idx_trn) = X(idx_trn);  % add known training values

      X_tst = ones(size(X))*NaN;
      X_tst(idx_tst) = X(idx_tst);  % add known training values

      % Predict the missing values here!
      tic;
      [~,~,~,X_pred] = ComputeBiases(X_trn, lam1, lam2);
      times(k) = toc;

      % Compute RMSE
      rmses(k) = RMSE(X_pred, X_tst, nil);  % error on known test values
    end

    fprintf('%d,%d,%f,%f,%f,%f\n', ...
            lam1, lam2, mean(rmses), std(rmses), mean(times), std(times));
  end
end

delete(pool);

