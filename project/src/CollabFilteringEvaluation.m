% Evaluation script for the collaborative filtering problem. Loads data
% matrix, splits known values into training and testing sets, and computes
% the MSE of the predicted to the true known entries of the test data.
%
% Loads data from Data.mat and calls PredictMissingValues.m.

% Setup
rand('seed', 1);  % fix random seed for reproducibility

% Constants
filename = '../data/Data.mat';
prc_trn = 0.5;  % percentage of training data
global nil;
nil = 0;  % missing value indicator

% Load data
L = load(filename);
X = L.X;

% Split intro training and testing index sets
idx = find(X ~= nil);
n = numel(idx);

n_trn = round(n*prc_trn);
rp = randperm(n);
idx_trn = idx(rp(1:n_trn));
idx_tst = idx(rp(n_trn+1:end));

% Build training and testing matrices
X_trn = ones(size(X))*nil;
X_trn(idx_trn) = X(idx_trn);  % add known training values

global X_tst;
X_tst = ones(size(X))*nil;
X_tst(idx_tst) = X(idx_tst);  % add known training values

% Predict the missing values here!
nils = X_trn == nil;
for k=2:20
  X_pred = PredictMissingValues(X_trn, nil, k);
    % X_pred = X_trn;
    % X_pred(nils) = NaN;
    % X_pred(nils) = nanmean(X_pred(:));

  % Compute MSE
  mse = sqrt(mean((X_tst(X_tst ~= nil) - X_pred(X_tst ~= nil)).^2));  % error on known test values

  fprintf('simple RSVD, 20 iter, r = 0.001, l = 0.02, K = %d; %f; -\n', k, mse);
  % fprintf('RMSE: %f\n', mse);
  %disp(['Root of Mean-squared error: ' num2str(mse)]);
end

