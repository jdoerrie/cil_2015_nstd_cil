function X_pred = IterSVD(X, f)
  % Implementation of iterative SVD

  nils = isnan(X);
  [~, ~, ~, B] = LearnBiases(X);
  X_imp = X - B;
  X_imp(nils) = 0;

  for iter=1:3
    X_imp = TruncatedSVD(X_imp, f);
    X_imp(~nils) = X(~nils) - B(~nils);
  end

  X_pred = X_imp + B;
end
