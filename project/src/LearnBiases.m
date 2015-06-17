function [mu, bu, bi, B] = LearnBiases(X, gamma, lambda)
  % set default values of lambda
  if (nargin < 2) gamma  = 1e-3; end
  if (nargin < 3) lambda = 1e-2; end

  is_local = true;

  if is_local
    nEpochs = Inf;
  else
    nEpochs = 40;
  end

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

  if is_local
    prev_rmse = Inf;
  end

  B = zeros(M,N);
  for iEpoch=1:nEpochs
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

    if is_local
      B = mu + bsxfun(@plus, bu, bi');
      curr_rmse = RMSE(B);
      if prev_rmse < curr_rmse
        break;
      end

      prev_rmse = curr_rmse;
    end
  end

  B = mu + bsxfun(@plus, bu, bi');
  if is_local
    fprintf('Epoch = %d, RMSE = %f, gamma = %f, lambda = %f\n', ...
      iEpoch, RMSE(B), gamma, lambda);
  end
end
