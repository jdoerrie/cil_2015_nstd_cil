function [mu, b_u, b_i, B] = LearnBiases(X, gamma, lambda)
  % set default values of lambda
  if (nargin < 2) gamma = 1e-3; end
  if (nargin < 3) lambda = 0.1; end

  % Learn Biases via Stochastic Gradient Descent.  Idea and notation are
  % inspired by "The BellKor Solution to the Netflix Grand Prize".
  [M, N] = size(X);

  mu = nanmean(X(:));
  % b_u are the user biases as a vector
  bu = zeros(M, 1);
  % b_i are the item biases as a vector
  bi = zeros(N, 1);

  % I and J are the index vectors corresponding to non-entries entries in X,
  % i.e. X(I(i), J(i)) is not NaN for all i
  [I, J] = find(~isnan(X));

  % nEntries is the total number of existing entries in the data matrix X
  nEntries = length(I);

  B = zeros(M,N);
  epoch = 0;
  while true
    epoch = epoch + 1;
    for idx=1:nEntries
      u = I(idx);
      i = J(idx);

      % compute current loss for given user/item combination
      err = X(u,i) - (mu + bu(u) + bi(i));
      % update both bias vectors depending on the current err, the regularize
      % parameter and the learning rate
      bu(u) = bu(u) + gamma * (err - lambda * bu(u));
      bi(i) = bi(i) + gamma * (err - lambda * bi(i));
    end

    B_curr = mu + bsxfun(@plus, bu, bi');
    if (RMSE(B) < RMSE(B_curr))
      break;
    end

    b_u = bu;
    b_i = bi;
    B = B_curr;
  end

  fprintf('LearnBiases: Gamma = %.3f, Lambda = %.3f, Epochs = %d, RMSE = %f\n', ...
    gamma, lambda, epoch, RMSE(B));
end

% function l = loss(X, b_u, b_i, lambda)
% l = norm2(X - bsxfun(@plus, b_u, b_i')) + lambda * (norm2(b_u) + norm2(b_i));
% end

% function n = norm2(X)
%   n = nansum(X(:).^2);
% end
