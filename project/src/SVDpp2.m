function X_pred = SVDpp2(X, K, gamma, lambda)
  % Implementation of SVD++ as seen in "Factorization Meets the
  % Neighborhood: a Multifaceted Collaborative Filtering Model"

  % Hyperparameters and default values
  if (nargin < 2)         K = 10; end % number of factors
  if (nargin < 3)  gamma = 0.010; end % learning rate
  if (nargin < 4) lambda = 0.100; end % regularizer term

  [~,~,~,X_pred] = ComputeBiases(X);
  % number of iterations over all known ratings per factor
  nEpochs = 50;
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

  X_prev = zeros(M,N);
  for epoch=1:nEpochs
    % Iterate over all known ratings
    for u=1:M
      Ru = R{u};
      pu = p(:,u);  % current latent factors for user u
      p2 = pu + isqrt(u) * sum(y(:,Ru),2);
      err_sum = zeros(K,1);
      for i=Ru
        qi = q(:,i);  % current latent factors for item i
        r_hat = X_pred(u,i) + p2' * qi;
        e_ui = X(u,i) - r_hat;

        err_sum = err_sum + e_ui*qi;
        q(:,i)  = qi    + gamma*( e_ui*p2 - lambda*qi );
      end

      p(:,u)  = pu    + gamma*( err_sum          - lambda*pu );
      y(:,Ru) = y(:,Ru) + gamma*( bsxfun(@minus, err_sum*isqrt(u), lambda*y(:,Ru)));
    end

    gamma = gamma * 0.85;
    % compute predictions
    p2 = zeros(K,M);
    for u=1:M
      Ru = R{u};
      p2(:,u) = isqrt(u) * sum(y(:,Ru),2);
    end

    X_curr = X_pred + (p + p2)' * q;
    X_curr = min(max(X_curr, 1), 5);
    fprintf('Epoch: %03d, Curr RMSE: %f, Gamma: %f\n', epoch, RMSE(X_curr), gamma);
    if (RMSE(X_curr) > RMSE(X_prev) - 1e-6)
      break;
    end

    X_prev = X_curr;
  end

  % p2 = zeros(M,1);
  % for u=1:M
  %   Ru = R{u};
  %   p2(u) = isqrt(u) * sum(y(Ru));
  % end

  % X_pred = X_pred + (p + p2)*q';
  X_pred = X_prev;
  X_pred = min(max(X_pred, 1), 5);
  fprintf('SVD++2, K = %d, gamma = %f, lambda = %f, rmse = %f\n', ...
    K, gamma, lambda, RMSE(X_pred));
end
