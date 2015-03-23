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

K = 5;
clusters = kmeans(X_pred, K);
means = zeros(K, 100);
for k = 1:K
    means(k,:) = nanmean(X(clusters == k, :));
end

