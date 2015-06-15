function [S, U] = ComputeSimilarity(X, nils, alpha, lambda8)

if nargin < 3
  alpha = 1;
end

if nargin < 4
  lambda8 = 100;
end

% Pearson correlation coefficient
rho = corr(X, 'rows', 'pairwise'); % 5.15 (Advances in CF)

S = rho .^ alpha;
S(1:size(S,1)+1:end) = -1; % (set diagonal to -1) don't take element itself into account for the neighbourhood

% TODO maybe consider 5.16 as a similarity measure?
% ratedElements = double(~nils);
% U = ratedElements' * ratedElements;
% UMinus1 = U - 1;
% S = (UMinus1 ./ (UMinus1 + lambda8)) .* rho; % 5.16
