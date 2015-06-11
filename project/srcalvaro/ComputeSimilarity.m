function [S, U] = ComputeSimilarity(X, nils, lambda8)

if nargin < 3
  lambda8 = 100;
end

XSquared = X.^2;

% empirical correlation coefficient
ro = (X' * X) / sqrt(XSquared' * XSquared); % 5.15 (Advances in CF)

ratedElements = double(~nils);
U = ratedElements' * ratedElements;
UMinus1 = U - 1;
S = (UMinus1 ./ (UMinus1 + lambda8)) .* ro; % 5.16
