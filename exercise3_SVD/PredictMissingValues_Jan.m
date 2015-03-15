function X_pred = PredictMissingValues(X, nil)
% Predict missing entries in matrix X based on known entries. Missing
% values in X are denoted by the special constant value nil.

% your collaborative filtering code here!

% replace nils with NaNs
nils = (X == nil);
X(nils) = NaN;

% try to approximate missing X(i,j) by mu + alpha(i) + beta(j),
% idea taken from http://stats.stackexchange.com/a/35476
Y = X;
mu = nanmean(Y(:));
Y = Y - mu;
beta = nanmean(Y);
Y = bsxfun(@minus, Y, beta);
alpha = nanmean(Y, 2);

means = bsxfun(@plus, alpha, beta) + mu;

% fill in nils with means
X_pred = X;
X_pred(nils) = means(nils);

% truncated SVD
[U, D, V] = svd(X_pred, 0);
k = 8;
res = U(:,1:k) * D(1:k,1:k) * V(:,1:k)';

% set nils to predicted values
X_pred(nils) = res(nils);
end
