function X_pred = PredictMissingValues(X, nil, k)
% Predict missing entries in matrix X based on known entries. Missing
% values in X are denoted by the special constant value nil.

% your collaborative filtering code here!

  if nargin < 2
    nil = 0;
  end

  % replace nils with NaNs
  if isnan(nil)
    nils = isnan(X);
  else
    nils = (X == nil);
    X(nils) = NaN;
  end

  if nargin < 3
    % best k for the new Data.mat
    k = 6;
  end

  X_pred = regSVD2(X, 8, 0.01, 0.02);
  X_pred(~nils) = X(~nils);
  X_pred = min(max(X_pred, 1), 5);
end
