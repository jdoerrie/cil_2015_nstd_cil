function X_pred = PredictMissingValues(X, nil, k)
% Predict missing entries in matrix X based on known entries. Missing
% values in X are denoted by the special constant value nil.

% your collaborative filtering code here!

if nargin < 2
  nil = 0;
end

if nargin < 3
  % best k for the new Data.mat
  k = 8;
end
% replace nils with NaNs
if isnan(nil)
  nils = isnan(X);
else
  nils = (X == nil);
  X(nils) = NaN;
end

% try to approximate missing b_ui by mu + b_u + b_i
% idea taken from "The BellKor Solution to the Netflix Grand Prize"
[mu, b_u, b_i] = GetBiases(X, NaN);

means = bsxfun(@plus, b_u, b_i) + mu;
% means might contain NaNs if a full row or column is missing
means(isnan(means)) = 0;

% fill in nils with means
X_pred = X;
X_pred(nils) = means(nils);

% truncated SVD
[U, D, V] = svd(X_pred, 0);
res = U(:,1:k) * D(1:k,1:k) * V(:,1:k)';

% set nils to predicted values
X_pred(nils) = res(nils);
end
