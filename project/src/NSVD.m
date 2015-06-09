function X_pred = NSVD(X, K, gamma, lambda_1, lambda_2)
  % Implementation of NSVD as seen in "Factorization Meets the Neighborhood: a
  % Multifaceted Collaborative Filtering Model" Section 2.3 and "Improving
  % regularized singular value decomposition for collaborative filtering"
  % Section 3.8

  % Hyperparameters and default values
  if (nargin < 2)           K = 6; end % number of factors
  if (nargin < 3)   gamma = 0.010; end % learning rate
  if (nargin < 4) lambda_1 = 0.10; end % regularizer term
  if (nargin < 5) lambda_2 = 0.10; end % regularizer term for biases

  % number of iterations over all known ratings per factor
  nEpochs = 15;
  % Dimensions of the input
  [M, N] = size(X);

  % U and I contain indices into X where ratings are available
  [U, I] = find(~isnan(X));

  % nRatings contains for every user the number of issued ratings
  nRatings = sum(~isnan(X), 2);

  % isqrt is the inverse sqrt of nRatings used for normalizing the sums
  % during the update steps
  isqrt = 1.0 ./ sqrt(max(nRatings, 1));

  % R contain for every user the indices into X where ratings are available.
  % Since the numbers of these indices are different for every user, we store
  % the result in a cell array.
  R = cell(M, 1);
  for i=1:M
    R{i} = find(~isnan(X(i,:)));
  end

  % initialize predictions
  X_pred = zeros(M,N);
  for k=1:K
    fprintf('Training Factor %d\n', k);
    mu = nanmean(X(:) - X_pred(:));
    bu = zeros(M,1);
    bi = zeros(N,1);

    % 0.1 to initialize is inspired by Simon Funk
    p = zeros(M,1);
    q = ones(N,1) * 0.1;
    x = zeros(N,1);

    for epoch=1:nEpochs
      % Iterate over all known ratings
      for u=1:M
        Ru = R{u};
        p(u) = sum(x(Ru)) * isqrt(u);
        err_sum = 0;
        pu = p(u);

        for i=Ru
          r_hat = X_pred(u,i) + mu + bu(u) + bi(i) + p(u) * q(i);
          e_ui = X(u,i) - r_hat;
          err_sum = err_sum + e_ui;

          puu = p(u);
          p(u)  = p(u)  + gamma*( e_ui      - lambda_1*p(u)  );
          bu(u) = bu(u) + gamma*( e_ui      - lambda_2*bu(u) );
          bi(i) = bi(i) + gamma*( e_ui      - lambda_2*bi(i) );
          q(i)  = q(i)  + gamma*( e_ui*puu  - lambda_1*q(i)  );
        end

        x(Ru) = x(Ru) + isqrt(u)*(p(u) - pu);
      end

      for u=1:M
        p(u) = isqrt(u) * sum(x(R{u}));
      end
      X_curr = X_pred + mu + bsxfun(@plus, bu, bi') + p*q';
      X_curr = min(max(X_curr, 1), 5);
      fprintf('Epoch: %03d, Curr RMSE: %f\n', epoch, RMSE(X_curr));
    end

    for u=1:M
      p(u) = isqrt(u) * sum(x(R{u}));
    end
    X_pred = X_pred + mu + bsxfun(@plus, bu, bi') + p*q';
    X_pred = min(max(X_pred, 1), 5);
    fprintf('NSVD, K = %d, gamma = %f, lambda = %f, rmse = %f\n', ...
      k, gamma, lambda_1, RMSE(X_pred));
  end
end
