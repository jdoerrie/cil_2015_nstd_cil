function m = reg_nanmean(x,dim,reg)
% Slightly modified version of the standard nanmean.  Takes another
% argument in form of a regularizer parameter that prevents div by 0 and
% shrinks towards zero.

nans = isnan(x);
x(nans) = 0;

if nargin < 3
  reg = eps;
end

% setting reg to at least epsilon makes sure that we will not divide by
% zero if there are only NaNs left
reg = max(eps, reg);

if nargin == 1 % let sum deal with figuring out which dimension to use
    n = sum(~nans) + reg;
    m = sum(x) ./ n;
else
    n = sum(~nans,dim) + reg;
    m = sum(x,dim) ./ n;
end
