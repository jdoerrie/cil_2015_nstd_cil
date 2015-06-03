function [mu, b_u, b_i] = GetBiases(X, nil, l_1, l_2)
% set default value of nil to 0
Y = X;
if nargin < 2
  nil = 0;
end

% set default values of l_1 and l_2 to eps
if nargin < 3
  l_1 = eps;
end

if nargin < 4
  l_2 = eps;
end

% set nil values to NaN
if ~isnan(nil)
  nils = (X == nil);
  X(nils) = NaN;
end

mu = reg_nanmean(X(:));
X = X - mu;

% b_i = reg_nanmean(X, 1, l_1);
% X = bsxfun(@minus, X, b_i);
% b_u = reg_nanmean(X, 2, l_2);
[b_u, b_i] = GetBiasesSGD(X, 300);
loss_mat = Y - bsxfun(@plus, b_u, b_i) - mu;
fprintf('sum loss: %f\n', nansum(loss_mat(:).^2));
end

function [b_u, b_i] = GetBiasesSGD(X, l)
[N, M] = size(X);
b_u = zeros(N, 1);
b_i = zeros(1, M);

[I, J] = find(~isnan(X));
fprintf('num non-NaNs: %d\n', length(I));
for iter=1:10000000
  idx = randi(size(I, 1));
  u = I(idx);
  i = J(idx);

  loss = X(u,i) - b_u(u) - b_i(i);
  b_u(u) = b_u(u) + (loss - l*b_u(u)) / 3000;
  b_i(i) = b_i(i) + (loss - l*b_i(i)) / 2000;
  if mod(iter, 100000) == 0
    fprintf('iter: %d\n', iter);
    loss_mat = X - bsxfun(@plus, b_u, b_i);
    fprintf('sum loss: %f\n', nansum(loss_mat(:).^2));
  end
end
end

function [b_u, b_i] = GetBiasesGD(X, l)
[N, M] = size(X);
b_u = zeros(N, 1);
b_i = zeros(1, M);

[I, J] = find(~isnan(X));
fprintf('num non-NaNs: %d\n', length(I));
for iter=1:1000
  for idx = 1:size(I, 1)
    u = I(idx);
    i = J(idx);

    loss = X(u,i) - b_u(u) - b_i(i);
    b_u(u) = b_u(u) + (loss - l*b_u(u)) / (iter);
    b_i(i) = b_i(i) + (loss - l*b_i(i)) / (iter);
  end
  if mod(iter, 10) == 0
    fprintf('iter: %d\n', iter);
    loss_mat = X - bsxfun(@plus, b_u, b_i);
    fprintf('sum loss: %f\n', nansum(loss_mat(:).^2));
  end
end
end
