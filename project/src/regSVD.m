function X_pred = regSVD(X, f, gamma, lambda)
  % Implementation of regularized SVD as seen in "Factorization Meets the
  % Neighborhood: a Multifaceted Collaborative Filtering Model" Section 2.3

  % Hyperparameters and default values
  if (nargin < 2)         f = 6; end % number of factors
  if (nargin < 3) gamma = 0.001; end % learning rate
  if (nargin < 4) lambda = 0.02; end % regularizer term

  % Dimensions of the input
  [M, N] = size(X);

  mu = nanmean(X(:)); % global mean
  bu = zeros(M, 1);   % user biases
  bi = zeros(N, 1);   % item biases
  p  = zeros(M, f);   % user latent factors
  q  = zeros(N, f);   % item latent factors

  % U and I contain indices into X where ratings are available
  [U, I] = find(~isnan(X));

  % initialize return value
  X_pred = zeros(M, N);

  % init error term
  old_err = RMSE(X_pred);

  % stopping criterion
  delta = 1e-4;

  % count epochs
  epoch = 0;

  while (epoch < 15)
    % Iterate over all known ratings
    epoch = epoch + 1;
    for idx=1 : length(U)
      if mod(idx, 1e5) == 0
        fprintf('epoch: %d, iter: %d\n', epoch, idx);
      end

      u = U(idx);             % current user
      i = I(idx);             % current item
      b = mu + bu(u) + bi(i); % baseline ratings for current user
      pu = p(u,:);            % current latent factors for user u
      qi = q(i,:);            % current latent factors for item i

      % approximation and error term
      r_hat = b + pu * qi';
      e_ui = X(u,i) - r_hat;

      % gradient updates
      bu(u)  = bu(u)  + gamma * ( e_ui      - lambda * bu(u)  );
      bi(i)  = bi(i)  + gamma * ( e_ui      - lambda * bi(i)  );
      p(u,:) = p(u,:) + gamma * ( e_ui * qi - lambda * p(u,:) );
      q(i,:) = q(i,:) + gamma * ( e_ui * pu - lambda * q(i,:) );
    end

    % compute predictions
    X_curr = mu + bsxfun(@plus, bu, bi') + p*q';
    new_err = RMSE(X_curr);
    fprintf('Curr RMSE: %f\n', new_err);
    if old_err < new_err
      break
    end

    X_pred = X_curr;
    if abs(old_err - new_err) < delta
      break;
    end

    new_err = old_err;
  end

  fprintf('rSVD, f = %d, gamma = %f, lambda = %f, rmse = %f\n', ...
      f, gamma, lambda, RMSE(X_pred));
end
