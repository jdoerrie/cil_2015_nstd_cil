function X_pred = CorNgbr(X)
  lambda = 100;
  k = 10;

  [M,N] = size(X);
  L = ~isnan(X);
  X0 = X;
  X0(isnan(X)) = 0;
  RHO = corr(X0);
  n = zeros(N);
  fprintf('Computing Similarities\n');
  for i=1:N
    for j=1:N
      n(i,j) = sum(L(:,i) & L(:,j));
    end
  end

  S = RHO .* n / (n + lambda);

  fprintf('Learning Biases\n');
  [~,~,~,B] = LearnBiases(X);
  X_pred = zeros(M,N);
  for u=1:M
    fprintf('Processing user %d\n', u);
    for i=1:N
      Sui = L(u,:) .* S(i,:);
      [Y,J] = sort(Sui, 'descend');
      Suik = Y(1:k);
      Juik = J(1:k);

      nom = (X(u,Juik) - B(u,Juik)) * Suik';
      den = sum(Suik);
      X_pred(u,i) = B(u,i) + nom / den;
    end
  end
end
