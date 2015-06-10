function X_pred = regSVD2(X, K, gamma, lambda)
  % Implementation of regularized SVD as seen in "Factorization Meets the
  % Neighborhood: a Multifaceted Collaborative Filtering Model" Section 2.3

  % Hyperparameters and default values
  if (nargin < 2)         K = 64; end % number of factors
  if (nargin < 3)  gamma = 0.050; end % learning rate
  if (nargin < 4) lambda = 0.050; end % regularizer term

  [mu,bu,bi,B] = LearnBiases(X);
  % number of iterations over all known ratings per factor
  nEpochs = 14;
  % Dimensions of the input
  [M, N] = size(X);

  % U and I contain indices into X where ratings are available
  [U, I] = find(~isnan(X));

  mu = nanmean(X(:));
  % initialize predictions
  UF = normrnd(0, 1e-2, K, M);
  IF = normrnd(0, 1e-2, K, N);

  % X_pred = zeros(M,N);
  for epoch=1:nEpochs
    % Iterate over all known ratings
    for idx=1 : length(U)
      u = U(idx); % current user
      i = I(idx); % current item

      r_hat = B(u,i) + UF(:,u)' * IF(:,i);
      e_ui = X(u,i) - r_hat;

      uF = UF(:,u);
      iF = IF(:,i);
      UF(:,u) = uF + gamma*( e_ui*iF - lambda*uF );
      IF(:,i) = iF + gamma*( e_ui*uF - lambda*iF );
    end

    gamma = gamma * 0.85;
    % compute predictions
    % X_curr = B + UF' * IF;
    % X_curr = min(max(X_curr, 1), 5);
    % fprintf('Epoch: %03d, Curr RMSE: %f, Gamma: %f\n', epoch, RMSE(X_curr), gamma);
    % if (RMSE(X_pred) < RMSE(X_curr))
    %   break;
    % else
    %   X_pred = X_curr;
    % end
  end

  X_pred = B + UF' * IF;
  X_pred = min(max(X_pred, 1), 5);
  fprintf('rSVD2, K = %d, gamma = %f, lambda = %f, rmse = %f\n', ...
    K, gamma, lambda, RMSE(X_pred));
end
