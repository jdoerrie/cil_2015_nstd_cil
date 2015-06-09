function [mu, b_u, b_i, B] = ComputeBiases(X)
mu = nanmean(X(:));
X = X - mu;

b_i = nanmean(X, 1);
b_i(isnan(b_i)) = 0;
X = bsxfun(@minus, X, b_i);
b_i = b_i';
b_u = nanmean(X, 2);
b_u(isnan(b_u)) = 0;

B = mu + bsxfun(@plus, b_u, b_i');
end
