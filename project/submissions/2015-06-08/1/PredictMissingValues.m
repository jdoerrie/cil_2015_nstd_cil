function X_pred = PredictMissingValues(X, nil)
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

X_pred = regSVD(X);
end
