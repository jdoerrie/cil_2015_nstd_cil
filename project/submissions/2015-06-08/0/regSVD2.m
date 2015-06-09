function X_pred = regSVD2(X, K, gamma, lambda)
  % Implementation of regularized SVD as seen in "Factorization Meets the
  % Neighborhood: a Multifaceted Collaborative Filtering Model" Section 2.3

  % Hyperparameters and default values
  if (nargin < 2)         K = 6; end % number of factors
  if (nargin < 3)  gamma = 0.01; end % learning rate
  if (nargin < 4) lambda = 0.02; end % regularizer term

  % Dimensions of the input
  [M, N] = size(X);

  [~,~,~,B] = LearnBiases(X);
  P = zeros(M, K);   % user latent factors
  Q = zeros(N, K);   % item latent factors

  % U and I contain indices into X where ratings are available
  [U, I] = find(~isnan(X));

  % initialize return value
  X_pred = zeros(M,N);

  for k=1:K
    % init error term
    P(:,k) = ones(M, 1) * 0.1;
    Q(:,k) = ones(N, 1) * 0.1;

    for epoch=1:10
      for idx=1 : length(U)
        u = U(idx);             % current user
        i = I(idx);             % current item
        pu = P(u,k);            % current latent factors for user u
        qi = Q(i,k);            % current latent factors for item i

        % approximation and error term
        r_hat = B(u,i) + P(u,:) * Q(i,:)';
        e_ui = X(u,i) - r_hat;

        % gradient updates
        P(u,k) = P(u,k) + gamma * ( e_ui * qi - lambda * P(u,k) );
        Q(i,k) = Q(i,k) + gamma * ( e_ui * pu - lambda * Q(i,k) );
      end
    end

    % compute predictions
    X_pred = B + P*Q';
  end
end
