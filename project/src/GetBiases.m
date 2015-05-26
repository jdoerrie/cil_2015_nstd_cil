function [mu, b_u, b_i] = GetBiases(X, nil, l_1, l_2)
if nargin < 2
  nil = 0;
end

if nargin < 3
  l_1 = eps;
end

if nargin < 4
  l_2 = eps;
end

if ~isnan(nil)
  nils = (X == nil);
  X(nils) = NaN;
end

mu = reg_nanmean(X(:));
X -= mu;

b_i = reg_nanmean(X, 1, l_1);
X = bsxfun(@minus, X, b_i);

b_u = reg_nanmean(X, 2, l_2);
end
