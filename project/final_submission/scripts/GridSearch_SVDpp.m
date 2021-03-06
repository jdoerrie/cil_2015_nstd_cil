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

pool = parpool('local', 10);

fprintf('Grid Search SVD++\n');
fprintf('K,Lambda,Gamma,Shrink,Mean_RMSE,Std_RMSE,Mean_CPU,Std_CPU\n');

range = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1];
shrinks = [0.750, 0.775, 0.800, 0.825, 0.850, 0.875, 0.900, 0.925, 0.950, 0.975, 1.000];

K = 64;

for lam=range
  for gam=range
    for shrink=shrinks
      parfor k=1:nFolds
      % for k=1:nFolds
        idx_trn = idxs_trn(k,:);
        idx_tst = idxs_tst(k,:);

        % Build training and testing matrices
        X_trn = ones(size(X))*NaN;
        X_trn(idx_trn) = X(idx_trn);  % add known training values

        X_tst = ones(size(X))*NaN;
        X_tst(idx_tst) = X(idx_tst);  % add known training values

        % Predict the missing values here!
        tic;
        X_pred = SVDpp(X_trn, K, lam, gam, shrink, X_tst, nil);
        times(k) = toc;

        % Compute RMSE
        rmses(k) = RMSE(X_pred, X_tst, nil);  % error on known test values
      end

      fprintf('%d,%f,%f,%f,%f,%f,%f,%f\n', ...
              K, lam, gam, shrink, mean(rmses), std(rmses), mean(times), std(times));
    end
  end
end

delete(pool);

