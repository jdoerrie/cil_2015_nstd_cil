function X_pred = SVDpp(X, K, gamma, lambda, shrink)
  % Implementation of SVD++ as seen in "Factorization Meets the
  % Neighborhood: a Multifaceted Collaborative Filtering Model"

  % Boolean to switch between local and judge mode
  is_local = false;

  % Hyperparameters and default values
  if (nargin < 2)         K = 64; end % number of factors
  if (nargin < 3)  gamma = 0.010; end % learning rate
  if (nargin < 4) lambda = 0.100; end % regularizer term
  if (nargin < 5) shrink = 1.000; end % shrinkage term

  orig_gamma = gamma;
  [~,~,~,B] = LearnBiases(X);
  X_n = (X - B);
  % number of iterations over all known ratings per factor
  nEpochs = 5;
  % Dimensions of the input
  [M, N] = size(X);

  % nRatings contains for every user the number of issued ratings
  nRatings = sum(~isnan(X), 2);

  % isqrt is the inverse sqrt of numRatings used for normalizing the sums
  % during the update steps
  isqrt = 1.0 ./ sqrt(max(nRatings, 1));

  % R contain for every user the indices into X where ratings are available.
  % Since the numbers of these indices are different for every user, we store
  % the result in a cell array.
  R = cell(M, 1);
  for i=1:M
    R{i} = find(~isnan(X(i,:)));
  end

  % [mu,bu,bi] = ComputeBiases(X - X_pred);

  % 0.1 to initialize is inspired by Simon Funk
  p  = randn(K,M) * 0.01;
  q  = randn(K,N) * 0.01;
  y  = randn(K,N) * 0.01;

  if is_local
    X_prev = zeros(M,N);
  end

  for epoch=1:nEpochs
    % Iterate over all known ratings
    for u=1:M
      Ru = R{u};
      pu = p(:,u);  % current latent factors for user u
      p2 = pu + isqrt(u) * sum(y(:,Ru),2);

      errors = X_n(u,Ru) - p2' * q(:,Ru);
      err_sum = q(:,Ru) * errors';
      q(:,Ru) = q(:,Ru) + gamma*(p2 * errors - lambda * q(:,Ru));
      p(:,u)  = pu      + gamma*( err_sum        - lambda*pu );
      y(:,Ru) = y(:,Ru) + gamma*( bsxfun(@minus, err_sum*isqrt(u), lambda*y(:,Ru)));
    end

    gamma = gamma * shrink;

    if is_local
      % compute predictions
      p2 = zeros(K,M);
      for u=1:M
        Ru = R{u};
        p2(:,u) = isqrt(u) * sum(y(:,Ru),2);
      end

      X_curr = B + (p + p2)' * q;
      X_curr = min(max(X_curr, 1), 5);
      % fprintf('Epoch: %02d\n', epoch);
      if (RMSE(X_curr) > RMSE(X_prev) - 1e-6)
        fprintf('Epoch: %03d, Curr RMSE: %f, Gamma: %f\n', epoch, RMSE(X_prev), gamma);
        break;
      end

      if (mod(epoch, 5) == 0)
        fprintf('Epoch: %03d, Curr RMSE: %f, Gamma: %f\n', epoch, RMSE(X_curr), gamma);
      end

      X_prev = X_curr;
    end
  end

  if is_local
    X_pred = X_prev;
    X_pred = min(max(X_pred, 1), 5);
    fprintf('SVD++, K = %d, gamma = %f, lambda = %f, shrinkage = %f, rmse = %f\n', ...
      K, orig_gamma, lambda, shrink, RMSE(X_pred));
  else
    for u=1:M
      Ru = R{u};
      p(:,u) = p(:,u) + isqrt(u) * sum(y(:,Ru),2);
    end

    X_pred = min(max(B + p' * q, 1), 5);
  end
end
