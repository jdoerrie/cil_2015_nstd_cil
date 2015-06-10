function X_pred = SVDpp2(X, K, gamma, lambda)
  % Implementation of regularized SVD as seen in "Factorization Meets the
  % Neighborhood: a Multifaceted Collaborative Filtering Model" Section 2.3

  % Hyperparameters and default values
  if (nargin < 2)         K = 64; end % number of factors
  if (nargin < 3)  gamma = 0.007; end % learning rate
  if (nargin < 4) lambda = 0.015; end % regularizer term

  [~,~,~,B] = ComputeBiases(X);
  % number of iterations over all known ratings per factor
  nEpochs = 100;
  % Dimensions of the input
  [M, N] = size(X);

  % U and I contain indices into X where ratings are available
  [U, I] = find(~isnan(X));

  % initialize predictions
  P = normrnd(0, 5e-2, K, M);
  Q = normrnd(0, 5e-2, K, N);
  Y = normrnd(0, 5e-2, K, N);
  NY = zeros(K, M);

  % nRatings contains for every user the number of issued ratings
  nRatings = sum(~isnan(X), 2);

  % isqrt is the inverse sqrt of numRatings used for normalizing the sums
  % during the update steps
  isqrt = 1.0 ./ sqrt(max(nRatings, 1));

  % R contain for every user the indices into X where ratings are available.
  % Since the numbers of these indices are different for every user, we store
  % the result in a cell array.
  R = cell(M, 1);
  for i=1:M
    R{i} = find(~isnan(X(i,:)));
  end

  X_pred = zeros(M,N);
  for epoch=1:nEpochs
    old_P = P;
    old_Q = Q;
    old_Y = Y;
    for u=1:M
      Ru = R{u};
      NY(:,u) = isqrt(u) * sum(Y(:,Ru), 2);

      for i=Ru
        pu = P(:,u);
        qi = Q(:,i);
        yi = Y(:,i);

        r_hat = B(u,i) + (NY(u) + pu)'*qi;
        e_ui = X(u,i) - r_hat;

        P(:,u) = pu + gamma*( e_ui*qi           - lambda*pu );
        Q(:,i) = qi + gamma*( e_ui*(NY(u) + pu) - lambda*qi );
        Y(:,i) = yi + gamma*( e_ui*qi           - lambda*yi );
      end
    end

    gamma = gamma * 0.90;
    % compute predictions
    for u=1:M
      Ru = R{u};
      NY(:,u) = isqrt(u) * sum(Y(:,Ru), 2);
    end

    X_curr = B + (P + NY)'*Q;
    X_curr = min(max(X_curr, 1), 5);
    fprintf('Epoch: %03d, Curr RMSE: %f, Gamma: %f\n', epoch, RMSE(X_curr), gamma);
%     if (RMSE(X_pred) <= RMSE(X_curr))
%       P = old_P;
%       Q = old_Q;
%       Y = old_Y;
%       gamma = gamma * 0.5;
%       lambda = lambda * 0.5;
%     else
%       gamma = gamma * 2.0;
%       lambda = lambda * 2.0;
       X_pred = X_curr;
%     end
  end

  % X_pred = B + UF' * IF;
  X_pred = min(max(X_pred, 1), 5);
  fprintf('SVD++2, K = %d, gamma = %f, lambda = %f, rmse = %f\n', ...
    K, gamma, lambda, RMSE(X_pred));
end
