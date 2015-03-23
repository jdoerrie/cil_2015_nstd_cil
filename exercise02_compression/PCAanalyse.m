function [U, lambda, mu, k] = PCAanalyse(X, pct)
mu = mean(X, 2);
if nargin == 1
    [U, ~, lambda] = pca(X');
else
    [U, ~, lambda, ~, V] = pca(X');
    k = find(cumsum(V) >= pct, 1);
end