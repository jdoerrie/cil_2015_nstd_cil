function X_pred = regSVD(X, K, gamma, lambda_1, lambda_2)
  % Implementation of regularized SVD as seen in "Factorization Meets the
  % Neighborhood: a Multifaceted Collaborative Filtering Model" Section 2.3

  % Hyperparameters and default values
  if (nargin < 2)          K = 64; end % number of factors
  if (nargin < 3)   gamma = 0.010; end % learning rate
  if (nargin < 4) lambda_1 = 0.10; end % regularizer term
  if (nargin < 5) lambda_2 = 0.10; end % regularizer term for biases

  minEpochs = 50;
  % number of iterations over all known ratings per factor
  nEpochs = 1e6;
  % Dimensions of the input
  [M, N] = size(X);

  % U and I contain indices into X where ratings are available
  [U, I] = find(~isnan(X));

  % initialize predictions
  [~,~,~,X_pred] = LearnBiases(X);
  for k=1:K
    fprintf('Training Factor %d\n', k);
    % [mu,bu,bi] = ComputeBiases(X - X_pred);

    % 0.1 to initialize is inspired by Simon Funk
    p  = randn(M,1) * 0.01;
    q  = randn(N,1) * 0.01;

    X_prev = zeros(M,N);
    for epoch=1:nEpochs
      % Iterate over all known ratings
      for idx=1 : length(U)
        u = U(idx); % current user
        i = I(idx); % current item
        pu = p(u);  % current latent factors for user u
        qi = q(i);  % current latent factors for item i

        % approximation and error term
        % r_hat = X_pred(u,i) + mu + bu(u) + bi(i) + pu * qi;
        r_hat = X_pred(u,i) + pu * qi;
        e_ui = X(u,i) - r_hat;

        % gradient updates
        % bu(u) = bu(u) + gamma*( e_ui    - lambda_2*bu(u) );
        % bi(i) = bi(i) + gamma*( e_ui    - lambda_2*bi(i) );
        p(u)  = pu    + gamma*( e_ui*qi - lambda_1*pu    );
        q(i)  = qi    + gamma*( e_ui*pu - lambda_1*qi    );
      end

      % compute predictions
      X_curr = X_pred + p*q';
      % X_curr = X_pred + mu + bsxfun(@plus, bu, bi') + p*q';
      X_curr = min(max(X_curr, 1), 5);
      if mod(epoch, 10) == 0
        fprintf('Epoch: %03d, Curr RMSE: %f\n', epoch, RMSE(X_curr));
      end
      if (RMSE(X_curr) > RMSE(X_prev) - 1e-6 && epoch > minEpochs)
        break;
      end

      X_prev = X_curr;
    end

    X_pred = X_pred + p*q';
    % X_pred = X_pred + mu + bsxfun(@plus, bu, bi') + p*q';
    X_pred = min(max(X_pred, 1), 5);
    fprintf('rSVD, K = %d, gamma = %f, lambda = %f, rmse = %f\n', ...
      k, gamma, lambda_1, RMSE(X_pred));
  end
end
