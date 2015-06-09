function X_pred = regSVD(X, K, gamma, lambda_1, lambda_2)
  % Implementation of regularized SVD as seen in "Factorization Meets the
  % Neighborhood: a Multifaceted Collaborative Filtering Model" Section 2.3

  % Hyperparameters and default values
  if (nargin < 2)           K = 6; end % number of factors
  if (nargin < 3)   gamma = 0.010; end % learning rate
  if (nargin < 4) lambda_1 = 0.10; end % regularizer term
  if (nargin < 5) lambda_2 = 0.10; end % regularizer term for biases

  % number of iterations over all known ratings per factor
  nEpochs = 100;
  % Dimensions of the input
  [M, N] = size(X);

  % U and I contain indices into X where ratings are available
  [U, I] = find(~isnan(X));

  % initialize predictions
  X_pred = zeros(M,N);
  for k=1:K
    mu = nanmean(X(:) - X_pred(:));
    bu = zeros(M,1);
    bi = zeros(N,1);

    % 0.1 to initialize is inspired by Simon Funk
    p  = ones(M,1) * 0.1;
    q  = ones(N,1) * 0.1;

    for epoch=1:nEpochs
      % Iterate over all known ratings
      for idx=1 : length(U)
        u = U(idx); % current user
        i = I(idx); % current item
        pu = p(u);  % current latent factors for user u
        qi = q(i);  % current latent factors for item i

        % approximation and error term
        e_ui = X(u,i) - (X_pred(u,i) + mu + bu(u) + bi(i) + pu*qi);

        % gradient updates
        bu(u) = bu(u) + gamma*( e_ui    - lambda_2*bu(u) );
        bi(i) = bi(i) + gamma*( e_ui    - lambda_2*bi(i) );
        p(u)  = pu    + gamma*( e_ui*qi - lambda_1*pu    );
        q(i)  = qi    + gamma*( e_ui*pu - lambda_1*qi    );
      end
    end

    X_pred = X_pred + mu + bsxfun(@plus, bu, bi') + p*q';
    X_pred = min(max(X_pred, 1), 5);
  end
end
