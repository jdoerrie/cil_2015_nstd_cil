function [S, U] = ComputeSimilarity(X, nils, lambda8)

if nargin < 3
  lambda8 = 100;
end

% Pearson correlation coefficient
rho = corr(X); % 5.15 (Advances in CF)

ratedElements = double(~nils);
U = ratedElements' * ratedElements;
UMinus1 = U - 1;
S = (UMinus1 ./ (UMinus1 + lambda8)) .* rho; % 5.16
