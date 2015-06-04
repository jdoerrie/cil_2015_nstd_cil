function [mu, b_u, b_i] = GetBiases(X, l_1, l_2)
% set default value of nil to 0
origX = X;

% set default values of l_1 and l_2 to eps
if nargin < 2
  l_1 = eps;
end

if nargin < 3
  l_2 = eps;
end

mu = reg_nanmean(X(:));
X = X - mu;

% b_i = reg_nanmean(X, 1, l_1);
% X = bsxfun(@minus, X, b_i);
% b_u = reg_nanmean(X, 2, l_2);
[b_u, b_i] = GetBiasesSGD(X, 0);
% loss_mat = origX - bsxfun(@plus, b_u, b_i) - mu;
% fprintf('sum loss: %f\n', nansum(loss_mat(:).^2));
end

function [b_u, b_i] = GetBiasesSGD(X, lambda)
% Learn Biases via Stochastic Gradient Descent.  Idea and notation are inspired
% by "The BellKor Solution to the Netflix Grand Prize".
[N, M] = size(X);
% b_u are the user biases as a column vector
b_u = zeros(N, 1);
% b_i are the item biases as a row vector
b_i = zeros(1, M);

% number of iterations of SGD
numIter = 1e7;

% number of iterations after which debug information is printed to the screen
numDbgIter = 1e6;
% user defined learn rate for users
u_lrate = 1/1000;

% user defined learn rate for items
i_lrate = 1/1000;

% I and J are the index vectors corresponding to non-entries entries in X,
% i.e. X(I(i), J(i)) is not NaN for all i
[I, J] = find(~isnan(X));

% numEntries is the total number of existing entries in the data matrix X
numEntries = length(I);
fprintf('num non-NaNs: %d\n', numEntries);

for iter=1:numIter
  idx = randi(numEntries);
  u = I(idx);
  i = J(idx);

  % compute current loss for given user/item combination
  loss = X(u,i) - b_u(u) - b_i(i);
  % update both bias vectors depending on the current loss, the regularize
  % parameter and the learning rate
  b_u(u) = b_u(u) + (loss - lambda*b_u(u)) * u_lrate;
  b_i(i) = b_i(i) + (loss - lambda*b_i(i)) * i_lrate;

  if mod(iter, numDbgIter) == 0
    fprintf('iter: %d\n', iter);
    loss_mat = X - bsxfun(@plus, b_u, b_i);
    fprintf('sum loss: %f\n', nansum(loss_mat(:).^2));
  end
end
end

% function [b_u, b_i] = GetBiasesGD(X, l)
% [N, M] = size(X);
% b_u = zeros(N, 1);
% b_i = zeros(1, M);

% [I, J] = find(~isnan(X));
% fprintf('num non-NaNs: %d\n', length(I));
% for iter=1:1000
%   for idx = 1:size(I, 1)
%     u = I(idx);
%     i = J(idx);

%     loss = X(u,i) - b_u(u) - b_i(i);
%     b_u(u) = b_u(u) + (loss - l*b_u(u)) / (iter);
%     b_i(i) = b_i(i) + (loss - l*b_i(i)) / (iter);
%   end
%   if mod(iter, 10) == 0
%     fprintf('iter: %d\n', iter);
%     loss_mat = X - bsxfun(@plus, b_u, b_i);
%     fprintf('sum loss: %f\n', nansum(loss_mat(:).^2));
%   end
% end
% end
