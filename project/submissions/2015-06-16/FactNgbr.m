function X_pred = FactNgbr(X, K, gamma, lambda, shrink)
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
  is_local = false;

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
  nEpochs = 25;

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

  % For each user determine the indexes into X where ratings are available.
  % This allows for fast batch processing of all issued ratings.  The
  % indexes need to be stored in a cell array because the number for each
  % user is different. size(R{i}) = [1,R]
  R = cell(M, 1);
  for i=1:M
    R{i} = find(~isnan(X(i,:)));
  end

  % Q(:,i), Y(:,i) and Z(:,i) are item latent and item factor vectors that
  % try to represent the item-item relations present in the date. All of
  % them are learned through a gradient descent algorithm and initialized
  % with a Gaussian distribution with mean 0 and standard deviation 0.01.
  Q = randn(K,N) * 0.01;
  Y = randn(K,N) * 0.01;
  Z = randn(K,N) * 0.01;

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

      % user u's known ratings, size(r_u) = [1,R]
      r_u = X(u,Ru);

      % the base line estimators for user u, size(B_u) = [1,R]
      B_u = B(u,Ru);

      QRu = Q(:,Ru);
      YRu = Y(:,Ru);
      ZRu = Z(:,Ru);

      % implied p(u) vector that is the normalized weighted sum of relevant
      % x and y weights. size(pu) = [K,1]
      pu = isqrt(u)*(YRu*(r_u - B_u)' + sum(ZRu, 2));

      % current approximation including baseline estimators and modified
      % latent vectors, size(rhat_u) = [1, R]
      rhat_u = B_u + pu'*QRu;

      % current error terms, size(e_u) = [R, 1]
      e_u = (r_u - rhat_u)';

      % Normalized error term multiplied with QRu used in updated of Y and
      % Z. size(Qe_u) = [K,1]
      Qe_u = isqrt(u)*QRu*e_u;

      % update latent item factor vectors, size(Q(:,Ru)) = [K, R]
      Q(:,Ru) = QRu + gamma*(pu*e_u' - lambda*QRu);

      % update item factor weighted vectors, size(Y(:,Ru) = [K, R]
      Y(:,Ru) = YRu + gamma*( Qe_u*(r_u - B_u) - lambda*YRu );

      % update current items factor vectors, size(Z(:,Ru)) = [K, R].
      % bsxfun is necessary, because size(Qe_u) = [K, 1] and we
      % want to subtract the regularizer term for every item in R(u).
      Z(:,Ru) = ZRu + gamma*( bsxfun(@minus, Qe_u, lambda*ZRu) );
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
      P = zeros(K,M);
      for u=1:M
        Ru = R{u};
        res_u = X(u,Ru) - B(u,Ru);
        P(:,u) = isqrt(u)*( Y(:,Ru)*res_u' + sum(Z(:,Ru), 2) );
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
      'FactNgbr, K = %d, gam = %f, lam = %f, shrink = %f, RMSE = %f\n', ...
       K, orig_gamma, lambda, shrink, RMSE(X_pred) ...
    );
  else
    P = zeros(K,M);
    for u=1:M
      Ru = R{u};
      res_u = X(u,Ru) - B(u,Ru);
      P(:,u) = isqrt(u)*( Y(:,Ru)*res_u' + sum(Z(:,Ru), 2) );
    end

    X_pred = B + P'*Q;
    X_pred = min(max(X_pred, 1), 5);
  end
end
