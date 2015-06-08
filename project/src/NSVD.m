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
  nEpochs = 5;
  % Dimensions of the input
  [M, N] = size(X);

  % U and I contain indices into X where ratings are available
  [U, I] = find(~isnan(X));

  % numRatings contains for every user the number of issued ratings
  numRatings = sum(~isnan(X), 2);

  % isqrt is the inverse sqrt of numRatings used for normalizing the sums
  % during the update steps
  isqrt = 1.0 ./ sqrt(numRatings + 1);

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
    q = ones(N,1) * 0.1;
    x = zeros(N,1);

    for epoch=1:nEpochs
      % Iterate over all known ratings
      for idx=1 : length(U)
        if mod(idx, 1e5) == 0
          fprintf('epoch: %d, iter: %d\n', epoch, idx);
        end
        u = U(idx); % current user
        i = I(idx); % current item
        qi = q(i);  % current latent factors for item i

        Ru = R{u};
        pu = sum(x(Ru)) * isqrt(u);
        % approximation and error term
        r_hat = X_pred(u,i) + mu + bu(u) + bi(i) + pu * qi;
        e_ui = X(u,i) - r_hat;

        % gradient updates
        bu(u) = bu(u) + gamma*( e_ui             - lambda_2*bu(u) );
        bi(i) = bi(i) + gamma*( e_ui             - lambda_2*bi(i) );
        x(Ru) = x(Ru) + gamma*( e_ui*qi*isqrt(u) - lambda_1*x(Ru) );
        q(i)  = qi    + gamma*( e_ui*pu          - lambda_1*qi    );
      end

      % compute predictions
      p = zeros(M,1);
      for u=1:M
        p(u) = isqrt(u) * sum(x(R{u}));
      end
      X_curr = X_pred + mu + bsxfun(@plus, bu, bi') + p*q';
      X_curr = min(max(X_curr, 1), 5);
      fprintf('Epoch: %03d, Curr RMSE: %f\n', epoch, RMSE(X_curr));
    end

    p = zeros(M,1);
    for u=1:M
      p(u) = isqrt(u) * sum(x(R{u}));
    end
    X_pred = X_pred + mu + bsxfun(@plus, bu, bi') + p*q';
    X_pred = min(max(X_pred, 1), 5);
    fprintf('NSVD, K = %d, gamma = %f, lambda = %f, rmse = %f\n', ...
      k, gamma, lambda_1, RMSE(X_pred));
  end
end
