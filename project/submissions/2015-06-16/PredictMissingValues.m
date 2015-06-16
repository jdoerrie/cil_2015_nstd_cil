function X_pred = PredictMissingValues(X, nil)
  % Predict missing entries in matrix X based on known entries. Missing
  % values in X are denoted by the special constant value nil.

  % your collaborative filtering code here!
%   nils = X_tst ~= nil;

  if nargin < 2
    nil = 0;
  end

  % replace nils with NaNs
  if isnan(nil)
    nans = isnan(X);
  else
    nans = (X == nil);
    X(nans) = NaN;
  end


%   X_pred = zeros(size(X));
%   X_pred = X_pred + 0.40 * SVDpp(X, 64, 0.02, 0.02, 0.775);
%   X_pred = X_pred + 0.55 * FactNgbr(X, 64, 0.01, 0.1, 0.775);
%   X_pred = X_pred + 0.05 * FactNgbrUser(X, 64, 0.01, 0.1, 0.975);

  [M, N] = size(X);
  X_preds = zeros(M, N, 7);

  % best params for 10 epochs
  % X_preds(:,:,1) = SVDpp(       X, 64, 0.02, 0.02, 0.775);
  % best params for 25 epochs
  X_preds(:,:,1) = SVDpp(       X, 64, 0.01, 0.05, 0.95);

  % best params for 10 and 25 epochs
  X_preds(:,:,2) = FactNgbr(    X, 64, 0.01, 0.1, 0.775);

  % best params for 10 epochs
  % X_preds(:,:,3) = FactNgbrUser(X, 64, 0.01, 0.1, 0.975);
  % best params for 5 epochs
  X_preds(:,:,3) = FactNgbrUser(X, 64, 0.02, 0.1, 0.975);

  % best params for 30 epochs
  X_preds(:,:,4) = IntModel(    X, 64, 0.005, 0.05, 0.975);

  X_preds(:,:,5) = DinaPCA(X);
  X_preds(:,:,6) = AlvaroGMM(X, NaN);
  X_preds(:,:,7) = 1;

  nBins  = 1000;
  lambda =  100;

%   range = [1 2 5]' * power(10, 0:9);
%   range = reshape(range, 1, numel(range));
%
%   bins = [1 2 5]' * power(10, 0:5);
%   bins = reshape(bins, 1, numel(bins));

  X_pred = BinnedRidgeRegression(X, X_preds, nBins, lambda);
%   for bin = bins
%       best_err = Inf;
%       best_lam = Inf;
%       for lam=range
%          err = RMSE(X_prev);
%          if err < best_err
%              best_err = err;
%              best_lam = lam;
%          end
%       end
%   end

  %   shrink = [0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 1.00];
%   for gam=range
%     for lam=range
%       for sh=shrink
%         X_pred = SVDpp(X, 64, gam, lam, sh);
%       end
%     end
%   end
end
