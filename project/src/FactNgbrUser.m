function X_pred = FactNgbrUser(X, K, gamma, lambda, shrink)
  % Matlab optimized implementation of a Factorized Neighborhood Model as
  % seen in "Factorization Meets the Neighborhood: a Multifaceted
  % Collaborative Filtering Model".  An approximation of the ratings is
  % achieved through the following formula:
  %
  % $$ \^{r}_{ui} = \mu + b_i + b_u + q_i^T \left( |R(u)|^{-\frac{1}{2}}
  % \sum_{j \in R(u)} (r_{uj} - b_{uj})y_j + z_j \right ) $$
  %
  % where \mu is the total average rating, b_i an item specific bias, b_u a
  % user specific bias, q_i an item specific latent factor vector and R(u)
  % the index set of ratings issued by a given user. y_j and z_j are item
  % specific factor vectors, so that users are also specified by the set of
  % items they rate.

  % Boolean to switch between local and judge mode. Local mode will print
  % debug statements and compute the current RMSE score after every
  % iteration through the data points.  Judge mode disables all of this,
  % resulting in a faster algorithm but no progress reporting during the
  % run.
  is_local = true;
  rng('default');

  % Hyperparameters and default values.
  if nargin < 2; K      =    64; end % number of latent factors
  if nargin < 3; gamma  = 0.005; end % learning rate
  if nargin < 4; lambda = 0.020; end % regularizer term
  if nargin < 5; shrink = 0.950; end % gamma shrinkage factor

  % Stores the original value of gamma to be able to log it later.  The
  % continued multiplication with the shrinkage term makes it impossible to
  % retrieve the original value otherwise.
  orig_gamma = gamma;

  % Precompute biases. Biases are not learned due to performance reasons.
  [mu, bu, bi, B] = ComputeBiases(X);

  % Define the number of epochs.  An epoch in this context is a complete
  % iteration over all present ratings in X.
  nEpochs = 5;

  % Determine the size of the input.  M is the number of users, N the
  % number of items.
  [M, N] = size(X);

  % Determine for each user the number of ratings he issued.
  % $$ nRatings \in R^M $$
  nRatings = sum(~isnan(X), 1);

  % Precompute for each user the inverse square root of the number of
  % ratings issued. This will be used to normalize sums in the gradient
  % update steps.
  % $$ iSqrt \in R^M $$
  iSqrt = 1.0 ./ sqrt(max(nRatings, 1));

  % For each user determine the indexes into X where ratings are available.
  % This allows for fast batch processing of all issued ratings.  The
  % indexes need to be stored in a cell array because the number for each
  % user is different. size(R{i}) = [R,1]
  R = cell(N,1);
  for i=1:N
    R{i} = find(~isnan(X(:,i)));
  end

  % Q(:,i), Y(:,i) and Z(:,i) are item latent and item factor vectors that
  % try to represent the item-item relations present in the date. All of
  % them are learned through a gradient descent algorithm and initialized
  % with a Gaussian distribution with mean 0 and standard deviation 0.01.
  P = randn(K,M) * 0.01;
  Z = randn(K,M) * 0.01;

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
    for i=1:N
      % indices of the ratings of item i
      Ri = R{i};

      % item i's known ratings, size(r_i) = [R,1]
      r_i = X(Ri,i);

      % the base line estimators for item i and its neightborhood,
      % size(B_i) = [R,1]
      B_i = B(Ri,i);


      % Prestored data for the current item, avoids unnecessary memory
      % accesses, size({P,Z}Ri) = [K,R]
      PRi = P(:,Ri);
      ZRi = Z(:,Ri);

      % implied p(u) vector that is the normalized weighted sum of relevant
      % x and y weights. size(qi) = [K,1]
      qi = iSqrt(i)*ZRi*(r_i - B_i);

      % current rating approximation including baseline estimators and
      % modified latent vectors, size(rhat_i) = [R,1]
      rhat_i = B_i + PRi'*qi;

      % current error terms, size(e_u) = [R,1]
      e_i = r_i - rhat_i;

      % Normalized error term multiplied with PRi used in update of Z.
      % size(Pe_i) = [K,1]
      Pe_i = iSqrt(i)*PRi*e_i;

      % update latent item factor vectors, size(P(:,Ri)) = [K, R]
      P(:,Ri) = PRi + gamma*( qi*e_i'           - lambda*PRi );

      % update item factor weighted vectors, size(Y(:,Ru) = [K, R]
      Z(:,Ri) = ZRi + gamma*( Pe_i*(r_i - B_i)' - lambda*ZRi );
    end

    % shrink gamma according to the specified factor
    gamma = gamma * shrink;

    % if running in local mode, compute the prediction for the current
    % iteration. P contains the modified user latent factors and gets
    % multiplied with Q. The current prediction is then B + P'*Q and gets
    % clamped to be within [1,5]. Then an RMSE score is computed and
    % printed to stdout every couple of iterations. If the score plus an
    % epsilon was worse than the previous iteration we stop the gradient
    % descent.
    if is_local
      Q = zeros(K,N);
      for i=1:N
        Ri = R{i};
        res_i = X(Ri,i) - B(Ri,i);
        Q(:,i) = iSqrt(i)*Z(:,Ri)*res_i;
      end

      X_curr = B + P'*Q;
      X_curr = min(max(X_curr, 1), 5);

      if (RMSE(X_curr) + 1e-6 > RMSE(X_prev))
        fprintf('Epoch: %03d, Curr RMSE: %f, Gamma: %f\n', iEpoch, ...
                RMSE(X_prev), gamma);
        break;
      end

      if (mod(iEpoch, 1) == 0)
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
      '%s, K = %d, gam = %f, lam = %f, shrink = %f, RMSE = %f\n', ...
      'FactNgbrUser', K, orig_gamma, lambda, shrink, RMSE(X_pred) ...
    );
  else
    Q = zeros(K,N);
    for i=1:N
      Ri = R{i};
      res_i = X(Ri,i) - B(Ri,i);
      Q(:,i) = iSqrt(i)*Z(:,Ri)*res_i;
    end

    X_pred = B + P'*Q;
    X_pred = min(max(X_pred, 1), 5);
  end
end
