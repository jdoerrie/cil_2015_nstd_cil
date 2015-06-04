function X_pred = PredictMissingValues(X, nil, k)
% Predict missing entries in matrix X based on known entries. Missing
% values in X are denoted by the special constant value nil.

% your collaborative filtering code here!

if nargin < 2
  nil = 0;
end

if nargin < 3
  % best k for the new Data.mat
  k = 6;
end
% replace nils with NaNs
if isnan(nil)
  nils = isnan(X);
else
  nils = (X == nil);
  X(nils) = NaN;
end
[P, Q, mu, bu, bi] = LearnVectors(X, nil, k, 0.01, 0);
X_pred = mu + bsxfun(@plus, bu, bi') + P*Q';
X_pred(~nils) = X(~nils);
X_pred = min(max(X_pred, 1), 5);

% try to approximate missing b_ui by mu + b_u + b_i
% idea taken from "The BellKor Solution to the Netflix Grand Prize"
% if isempty(mu)
%   [mu, b_u, b_i] = GetBiases(X, NaN);
% end

% means = bsxfun(@plus, b_u, b_i) + mu;
% % means might contain NaNs if a full row or column is missing
% means(isnan(means)) = 0;

% subtract means from X
% X_norm = X - means;
% X_norm(nils) = 0;

% truncated SVD
% [U, D, V] = svd(X_norm, 0);
% X_norm = U(:,1:k) * D(1:k,1:k) * V(:,1:k)';
% X_norm = X_norm + means;

% X_pred = X;
% X_pred(nils) = X_norm(nils);
% fill in nils with means
% X_pred = X;
% X_pred(nils) = means(nils);

% % truncated SVD
% [U, D, V] = svd(X_pred, 0);
% res = U(:,1:k) * D(1:k,1:k) * V(:,1:k)';

% % set nils to predicted values
% X_pred(nils) = res(nils);
end
