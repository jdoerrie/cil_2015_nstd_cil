function X_pred = regSVD2(X, K, gamma, lambda, X_tst, nil)
  % Implementation of regularized SVD as seen in "Factorization Meets the
  % Neighborhood: a Multifaceted Collaborative Filtering Model" Section
  % 2.3. This is not used in the final submission and just included because
  % it served as a base line.

  % Hyperparameters and default values
  if (nargin < 2)         K = 64; end % number of factors
  if (nargin < 3)  gamma = 0.050; end % learning rate
  if (nargin < 4) lambda = 0.050; end % regularizer term

  [mu,bu,bi,B] = ComputeBiases(X);
  % number of iterations over all known ratings per factor
  nEpochs = 25;
  % Dimensions of the input
  [M, N] = size(X);

  % For each user determine the indices into X where ratings are available.
  % This allows for fast batch processing of all issued ratings.  The
  % indices need to be stored in a cell array because the number for each
  % user is different.
  % $$ R{i} \in R^{R_i} $$
  R = cell(M, 1);
  for i=1:M
    R{i} = find(~isnan(X(i,:)));
  end

  % U and I contain indices into X where ratings are available
  [U, I] = find(~isnan(X));

  mu = nanmean(X(:));
  % initialize predictions
  UF = normrnd(0, 1e-2, K, M);
  IF = normrnd(0, 1e-2, K, N);

  prev_rmse = Inf;
  for epoch=1:nEpochs
    % Iterate over all known ratings
    for u=1:M
      for i=R{u}
        r_hat = B(u,i) + UF(:,u)' * IF(:,i);
        e_ui = X(u,i) - r_hat;

        uF = UF(:,u);
        iF = IF(:,i);
        UF(:,u) = uF + gamma*( e_ui*iF - lambda*uF );
        IF(:,i) = iF + gamma*( e_ui*uF - lambda*iF );
      end
    end

    curr_rmse = RMSE(B + UF' * IF, X_tst, nil);
    if prev_rmse < curr_rmse + 1e-4
      break;
    end

    prev_rmse = curr_rmse;
    gamma = gamma * 0.85;
  end

  X_pred = B + UF' * IF;
  X_pred = min(max(X_pred, 1), 5);
  fprintf('regSVD, Epochs = %d, K = %d, gamma = %f, lambda = %f, rmse = %f\n', ...
    epoch, K, gamma, lambda, curr_rmse);
end
