function X_pred = AsymmetricSVD(X, K)
  % Implementation of Asymmetric SVD as seen in "Factorization Meets the
  % Neighborhood: a Multifaceted Collaborative Filtering Model" Section 4

  % Hyperparameters
  gamma = 0.02; % learning rate
  lambda = 0.04; % regularizer term

  % Dimensions of the input
  [M, N] = size(X);

  [~,~,~,B] = LearnBiases(X);
  q = zeros(N, K);     % item latent factors
  x = zeros(N, K);     % explicit weight
  y = zeros(N, K);     % implicit weight

  % U and I contain indices into X where ratings are available
  [U, I] = find(~isnan(X));

  % numRatings contains for every user the number of issued ratings
  numRatings = sum(~isnan(X), 2);

  % isqrt is the inverse sqrt of numRatings used for normalizing the sums
  % during the update steps
  isqrt = 1.0 ./ sqrt(numRatings);

  % Js contain for every user the indices into X where ratings are available.
  % Since the numbers of these indices are different for every user, we store
  % the result in a cell array.
  Js = cell(M, 1);
  for i=1:M
    Js{i} = find(~isnan(X(i,:)));
  end

  % initialize return value
  X_pred = B;
  old_err = RMSE(X_pred);
  fprintf('Base RMSE: %f\n', old_err);
  my_eps = 1e-4;

  for k=1:K
  for epoch=1:10
    % Iterate over all known ratings
    for idx=1 : length(U)
      if mod(idx, 1e5) == 0
        fprintf('epoch: %d, iter: %d\n', epoch, idx);
      end

      u = U(idx);            % current user
      i = I(idx);            % current item
      Bu = mu + bu(u) + bi'; % baseline ratings for current user
      qi = q(i,:);           % current latent factors for item i

      J = Js{u};                                    % read indices
      sum_x = isqrt(u) * (X(u,J) - Bu(J)) * x(J,:); % take weighted sum of x's
      sum_y = isqrt(u) * sum(y(J,:));               % take sum of relevant y's

      % approximation and error term
      r_hat = Bu(i) + qi * (sum_x' + sum_y');
      e_ui = X(u,i) - r_hat;

      % gradient updates
      bu(u)  = bu(u)  + gamma * ( e_ui                   - lambda * bu(u)  );
      bi(i)  = bi(i)  + gamma * ( e_ui                   - lambda * bi(i)  );
      q(i,:) = qi     + gamma * ( e_ui * (sum_x + sum_y) - lambda * qi     );
      x(J,:) = x(J,:) + gamma * ( e_ui * isqrt(u) * ...
          bsxfun(@times, qi, (X(u,J) - Bu(J))')          - lambda * x(J,:) );
      y(J,:) = y(J,:) + gamma * ( bsxfun(@minus, e_ui * isqrt(u) * qi, lambda * y(J,:)) );
    end

    % compute implied user vectors
    p = zeros(M, f);
    for u=1:M
      J = Js{u};
      sum_x = isqrt(u) * (X(u,J) - Bu(J)) * x(J,:);
      sum_y = isqrt(u) * sum(y(J,:));
      p(u,:) = sum_x + sum_y;
    end

    % compute predictions
    X_curr = mu + bsxfun(@plus, bu, bi') + p*q';
    new_err = RMSE(X_curr);
    fprintf('Curr RMSE: %f\n', new_err);
    if old_err < new_err
      break
    end

    X_pred = X_curr;
    if abs(old_err - new_err) < my_eps
      break;
    end

    new_err = old_err;
  end
end
