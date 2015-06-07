function X_pred = IterSVD(X, f)
  % Implementation of iterative SVD as seen in "Factorization Meets the
  % Neighborhood: a Multifaceted Collaborative Filtering Model" Section 2.3

  nils = isnan(X);
  [~, ~, ~, B] = LearnBiases(X);
  X_imp = X - B;
  X_imp(nils) = 0;
  old_err = RMSE(X_imp + B);
  X_pred = X_imp + B;
  delta = 1e-4;

  iter = 0;
  while (true)
    iter = iter + 1;
    X_imp = TruncatedSVD(X_imp, f);
    X_imp(~nils) = X(~nils) - B(~nils);
    new_err = RMSE(X_imp + B);
    fprintf('iter: %d, err = %f\n', iter, new_err);
    if (old_err < new_err)
      break
    end

    X_pred = X_imp + B;
    if (abs(new_err - old_err) < delta)
      break
    end

    old_err = new_err;
  end

  fprintf('IterSVD: f = %d, rmse = %f\n', f, RMSE(X_pred));
end
