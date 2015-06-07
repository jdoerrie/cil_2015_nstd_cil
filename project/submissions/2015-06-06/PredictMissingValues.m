function X_pred = PredictMissingValues(X, nil, k)
if nargin < 2
  nil = 0;
end
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
k = 7;
lambda = 0.1;
[mu, bu, bi] = LearnBiases(X, lambda);
BaseLine = mu + bsxfun(@plus, bu, bi');
X_pred = X - BaseLine;
X_pred(nils) = 0;
X_pred = TruncatedSVD(X_pred, k);
X_pred = X_pred + BaseLine;
X_pred = min(max(X_pred, 1), 5);
end
