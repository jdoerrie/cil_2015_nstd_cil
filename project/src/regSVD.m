function X_pred = regSVD(X, K, gamma, lambda_1, lambda_2)
  % Implementation of regularized SVD as seen in "Factorization Meets the
  % Neighborhood: a Multifaceted Collaborative Filtering Model" Section 2.3

  % Hyperparameters and default values
  if (nargin < 2)         K = 6; end % number of factors
  if (nargin < 3)  gamma = 0.01; end % learning rate
  if (nargin < 4) lambda_1 = 0.02; end % regularizer term
  if (nargin < 5) lambda_2 = 0.10; end % regularizer term for biases

  % Dimensions of the input
  [M, N] = size(X);

  mu = zeros(1, K); % global bias per factor
  bu = zeros(M, K); % user biases per factor
  bi = zeros(N, K); % item biases per factor
  P  = zeros(M, K); % user latent factors
  Q  = zeros(N, K); % item latent factors

  % U and I contain indices into X where ratings are available
  [U, I] = find(~isnan(X));

  % initialize return value
  X_pred = zeros(M,N);

  % stopping criterion
  delta = 1e-4;

  for k=1:K
    % init error term
    old_err = Inf;

    P(:,k) = ones(M, 1) * 0.1;
    Q(:,k) = ones(N, 1) * 0.1;
    fprintf('Training Feature %d\n', k);

    mu(k) = nanmean(X(:));
    for epoch=1:10
      % Iterate over all known ratings
      for idx=1 : length(U)
        if mod(idx, 1e5) == 0
          fprintf('epoch: %d, iter: %d\n', epoch, idx);
        end

        u = U(idx);             % current user
        i = I(idx);             % current item
        pu = P(u,k);            % current latent factors for user u
        qi = Q(i,k);            % current latent factors for item i

        % approximation and error term
        r_hat = mu(k) + bu(u,k) + bi(i,k) + P(u,k) * Q(i,k);
        e_ui = X(u,i) - r_hat;

        % gradient updates
        bu(u,k) = bu(u,k) + gamma * ( e_ui      - lambda_2 * bu(u,k) );
        bi(i,k) = bi(i,k) + gamma * ( e_ui      - lambda_2 * bi(i,k) );
        P(u,k)  = P(u,k)  + gamma * ( e_ui * qi - lambda_1 * P(u,k)  );
        Q(i,k)  = Q(i,k)  + gamma * ( e_ui * pu - lambda_1 * Q(i,k)  );
      end

      % compute predictions
      X_curr = getRatings(mu,bu,bi,P,Q);
      new_err = RMSE(X_curr);
      fprintf('Curr RMSE: %f\n', new_err);
      if old_err < new_err
        %break
      end

      X_pred = X_curr;
      if abs(old_err - new_err) < delta
        %break;
      end

      old_err = new_err;
    end

    X = X - (mu(k) + bsxfun(@plus, bu(:,k), bi(:,k)') + P(:,k) * Q(:,k)');
    fprintf('rSVD, K = %d, gamma = %f, lambda = %f, rmse = %f\n', ...
      k, gamma, lambda_1, RMSE(X_pred));
  end
end

function X_pred = getRatings(mu, bu, bi, P, Q)
  M = size(P, 1);
  N = size(Q, 1);
  K = size(P, 2);
  X_pred = zeros(M,N);
  for k=1:K
    X_pred = X_pred + mu(k) + bsxfun(@plus, bu(:,k), bi(:,k)') + P(:,k) * Q(:,k)';
    X_pred = min(max(X_pred, 1), 5);
  end
end
