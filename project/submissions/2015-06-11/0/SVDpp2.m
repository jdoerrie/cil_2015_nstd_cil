function X_pred = SVDpp2(X, K, gamma, lambda)
  % Implementation of SVD++ as seen in "Factorization Meets the
  % Neighborhood: a Multifaceted Collaborative Filtering Model"

  % Hyperparameters and default values
  if (nargin < 2)         K = 64; end % number of factors
  if (nargin < 3)  gamma = 0.010; end % learning rate
  if (nargin < 4) lambda = 0.100; end % regularizer term

  [~,~,~,B] = ComputeBiases(X);
  X_n = (X - B);
  % number of iterations over all known ratings per factor
  nEpochs = 25;
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

  % 0.1 to initialize is inspired by Simon Funk
  p  = randn(K,M) * 0.01;
  q  = randn(K,N) * 0.01;
  y  = randn(K,N) * 0.01;

  for epoch=1:nEpochs
    % Iterate over all known ratings
    for u=1:M
      Ru = R{u};
      pu = p(:,u);  % current latent factors for user u
      p2 = pu + isqrt(u) * sum(y(:,Ru),2);

      errors = X_n(u,Ru) - p2' * q(:,Ru);
      err_sum = q(:,Ru) * errors';
      p(:,u)  = pu    + gamma*( err_sum        - lambda*pu );
      q(:,Ru) = q(:,Ru) + gamma * (p2 * errors - lambda * q(:,Ru));
      y(:,Ru) = y(:,Ru) + gamma*( bsxfun(@minus, err_sum*isqrt(u), lambda*y(:,Ru)));
    end
  end

  for u=1:M
    Ru = R{u};
    p(:,u) = p(:,u) + isqrt(u) * sum(y(:,Ru), 2);
  end

  X_pred = B + p' * q;
  X_pred = min(max(X_pred, 1), 5);
end
