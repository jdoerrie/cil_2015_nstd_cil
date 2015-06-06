function [mu, b_u, b_i] = ComputeBiases(X, lambda)
mu = mean(X(:), 'omitnan');
X = X - mu;

b_i = mean(X, 1, 'omitnan');
b_i(isnan(b_i)) = 0;
X = bsxfun(@minus, X, b_i);
b_i = b_i';
b_u = mean(X, 2, 'omitnan');
b_u(isnan(b_u)) = 0;
end
