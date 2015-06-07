function [mu, b_u, b_i, B] = LearnBiases(X, lambda)
% set default values of lambda
if nargin < 2
  lambda = 0.1;
end

[mu, b_u, b_i] = LearnBiasesSGD(X, lambda);
B = mu + bsxfun(@plus, b_u, b_i');
end

function [mu, b_u, b_i] = LearnBiasesSGD(X, lambda)
mu = nanmean(X(:));
X = X - mu;

% Gamma is the learning rate
gamma = 0.001;

my_eps = 1e2;
% Learn Biases via Stochastic Gradient Descent.  Idea and notation are inspired
% by "The BellKor Solution to the Netflix Grand Prize".
[M, N] = size(X);
% b_u are the user biases as a vector
b_u = zeros(M, 1);
% b_i are the item biases as a vector
b_i = zeros(N, 1);

% I and J are the index vectors corresponding to non-entries entries in X,
% i.e. X(I(i), J(i)) is not NaN for all i
[I, J] = find(~isnan(X));

% numEntries is the total number of existing entries in the data matrix X
numEntries = length(I);

best_loss = loss(X, b_u, b_i, lambda);
while true
  bu = b_u;
  bi = b_i;
  for idx=1:numEntries
    u = I(idx);
    i = J(idx);

    % compute current loss for given user/item combination
    err = X(u,i) - bu(u) - bi(i);
    % update both bias vectors depending on the current err, the regularize
    % parameter and the learning rate
    bu(u) = bu(u) + gamma * (err - lambda * bu(u));
    bi(i) = bi(i) + gamma * (err - lambda * bi(i));
  end

  curr_loss = loss(X, bu, bi, lambda);
  if best_loss < curr_loss
    break;
  end
  b_u = bu;
  b_i = bi;
  if abs(curr_loss - best_loss) < my_eps
    break;
  end
  best_loss = curr_loss;
end
end

function l = loss(X, b_u, b_i, lambda)
l = norm2(X - bsxfun(@plus, b_u, b_i')) + lambda * (norm2(b_u) + norm2(b_i));
end
