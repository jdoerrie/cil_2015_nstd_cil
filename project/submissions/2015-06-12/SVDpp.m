function X_pred = SVDpp(X, K, gamma, lambda, shrink)
  % Matlab optimized implementation of SVD++ as seen in "Factorization
  % Meets the Neighborhood: a Multifaceted Collaborative Filtering Model".
  % An approximation of the ratings is achieved through the following
  % formula:
  %
  % $$ \^{r}_{ui} = \mu + b_i + b_u + q_i^T \left( p_u +
  % |R(u)|^{-\frac{1}{2}} \sum_{j \in R(u)} y_j \right ) $$
  %
  % where \mu is the average rating, b_i an item specific bias, b_u
  % a user specific bias, q_i an item specific latent factor vector, p_u a
  % user specific latent factor vector and R(u) the index set of ratings
  % issued by a given user. y_j is an item specific factor vector, so that
  % users are also specified by the set of items they rate.

  % Boolean to switch between local and judge mode. Local mode will print
  % debug statements and compute the current RMSE score after every
  % iteration through the data points.  Judge mode disables all of this,
  % resulting in a faster algorithm but no progress reporting during the
  % run.
  is_local = false;

  % Hyperparameters and default values optimized via cross validation.
  if nargin < 2; K      =    64; end % number of latent factors
  if nargin < 3; gamma  = 0.005; end % learning rate
  if nargin < 4; lambda = 0.020; end % regularizer term
  if nargin < 5; shrink = 0.950; end % gamma shrinkage factor

  % Stores the original value of gamma to be able to log it later.  The
  % continued multiplication with the shrinkage term makes it impossible to
  % retrieve the original value otherwise.
  orig_gamma = gamma;

  % Boolean flag indicating whether biases should be learned during the
  % gradient descent or whether just precomputed values should be used.
  % In either case biases are computed, since they will serve as initial
  % values even when learned later on.
  learn_biases = false;

  [mu, bu, bi, B] = ComputeBiases(X);

  % X "normalized".  Stores the residuals after subtracting the biases from
  % X.  The remaining code tries to approximate these residuals as good as
  % possible.
  if ~learn_biases
    X_n = X - B;
  end

  % Define the number of epochs.  An epoch in this context is a complete
  % iteration over all present ratings in X.
  nEpochs = 30;

  % Determine the size of the input.  M is the number of users, N the
  % number of items.
  [M, N] = size(X);

  % Determine for each user the number of ratings he issued.
  % $$ nRatings \in R^M $$
  nRatings = sum(~isnan(X), 2);

  % Precompute for each user the inverse square root of the number of
  % ratings issued. This will be used to normalize sums in the gradient
  % update steps.
  % $$ isqrt \in R^M $$
  isqrt = 1.0 ./ sqrt(max(nRatings, 1));

  % For each user determine the indices into X where ratings are available.
  % This allows for fast batch processing of all issued ratings.  The
  % indices need to be stored in a cell array because the number for each
  % user is different.
  % $$ R{i} \in R^{R_i} $$
  R = cell(M, 1);
  for i=1:M
    R{i} = find(~isnan(X(i,:)));
  end

  % P(:,u), Q(:,i) and Y(:,i) are user latent, item latent and item factor
  % vectors. All of them are learned through a gradient descent algorithm
  % and initialized with a gaussian distribution with mean 0 and standard
  % deviation 0.01.
  P = randn(K,M) * 0.01;
  Q = randn(K,N) * 0.01;
  Y = randn(K,N) * 0.01;

  % If local mode is enabled we keep track of the previous iteration's
  % ratings approximation to be able to measure the difference in RMSE
  % between two consecutive iterations.
  if is_local
    X_prev = zeros(M,N);
  end

  % Actual implementation of Gradient Descent.  We aim to minize the
  % following objective function:
  %
  % $$ \sum_{(u,i) \in \mathcal{K}} r_{ui} - \^{r}_{ui} + lambda *
  %       ( ||q_i||^2 + ||p_u||^2 + |R(u)|^{-1/2}\sum_{j \in R(u)} )
  % ||y_j||^2 $$ leading to the following partial derivatives and update
  % rules for a given datapoint (u,i), where e_ui = r_ui - ^r_ui.
  %
  % q_i += gamma*( e_ui*( p_u + |R(u)|^{-1/2} \sum_{j \in R(u) y_j} )
  %                                       - lambda*q_i )
  % p_u += gamma*( e_ui*q_i               - lambda*p_u )
  % \forall j \in R(u):
  % y_j += gamma*( e_ui*|R(u)|^{-1/2}*q_i - lambda*y_j )
  %
  % For efficiency reasons we will not update after each processed data
  % point, but do a batch process after processing all rated items for a
  % given user.

  for iEpoch=1:nEpochs
    % Iterate over all known ratings
    for u=1:M
      % indices of the ratings of user u
      Ru = R{u};

      % modified p(u) vector that takes the normalized sum of the y weights
      % into account
      pu = P(:,u) + isqrt(u) * sum(Y(:,Ru), 2);

      % user u's known ratings
      r_u = X(u,Ru);

      % current aproximation including baseline estimators and modified
      % latent vectors
      rhat_u = B(u,Ru) + pu' * Q(:,Ru);

      % current error terms, size(e_u) = [R, 1]
      e_u = (r_u - rhat_u)';

      % update current user latent vector, size(P(:,u)) = [K, 1]
      P(:,u)  = P(:,u)  + gamma*( Q(:,Ru)*e_u - lambda*P(:,u) );

      % update current items factor vectors, size(Y(:,Ru)) = [K, R].
      % bsxfun is necessary, because size(Q(:,Ru) * e_u) = [K, 1] and we
      % want to subtract the regularizer term for every item in R(u).
      Y(:,Ru) = Y(:,Ru) + gamma* ...
        ( bsxfun(@minus, isqrt(u)*Q(:,Ru)*e_u, lambda*Y(:,Ru)) );

      % update latent item factor vectors, size(Q(:,Ru)) = [K, R]
      Q(:,Ru) = Q(:,Ru) + gamma*(pu*e_u' - lambda * Q(:,Ru));
    end

    % shrink gamma according to the specified factor
    gamma = gamma * shrink;

    % if running in local mode, compute the prediction for the current
    % iteration. V contains the modifed user latent factors and gets
    % multiplied with Q. The current prediction is then B + Q'*P and gets
    % clamped to be within [1,5]. Then an RMSE score is computed and
    % printed to stdout every couple of iterations. If the score plus an
    % epsilon was worse than the previous iteration we stop the gradient
    % descent.
    if is_local
      V = zeros(K,M);
      for u=1:M
        Ru = R{u};
        V(:,u) = P(:,u) + isqrt(u) * sum(Y(:,Ru),2);
      end

      X_curr = B + V'*Q;
      X_curr = min(max(X_curr, 1), 5);

      if (RMSE(X_curr) + 1e-6 > RMSE(X_prev))
        fprintf('Epoch: %03d, Curr RMSE: %f, Gamma: %f\n', iEpoch, ...
                RMSE(X_prev), gamma);
       break;
      end

      if (mod(iEpoch, 5) == 0)
        fprintf('Epoch: %03d, Curr RMSE: %f, Gamma: %f\n', iEpoch, ...
                RMSE(X_curr), gamma);
      end

      X_prev = X_curr;
    end
  end

  % This part assigns the computed predictions to the output variable.  If
  % running in local mode this is equivalent to the last X_prev, so it
  % simply assigns the value and again prints the parameters plus RMSE
  % score.  If running in judge mode only the predictions are computed and
  % returned.
  if is_local
    X_pred = X_prev;
    fprintf(...
      'SVD++, K = %d, gam = %f, lam = %f, shrink = %f, RMSE = %f\n', ...
       K, orig_gamma, lambda, shrink, RMSE(X_pred) ...
    );
  else
    for u=1:M
      Ru = R{u};
      P(:,u) = P(:,u) + isqrt(u)*sum(Y(:,Ru), 2);
    end

    X_pred = B + P'*Q;
    X_pred = min(max(X_pred, 1), 5);
  end
end
