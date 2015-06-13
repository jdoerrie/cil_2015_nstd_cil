function X_pred = PredictMissingValues(X, nil)
  % Predict missing entries in matrix X based on known entries. Missing
  % values in X are denoted by the special constant value nil.

  % your collaborative filtering code here!

  if nargin < 2
    nil = 0;
  end

  % replace nils with NaNs
  if isnan(nil)
    nils = isnan(X);
  else
    nils = (X == nil);
    X(nils) = NaN;
  end

  range = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1];
  shrink = [0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 1.00];
  for gam=range
    for lam=range
      for sh=shrink
        X_pred = SVDpp(X, 64, gam, lam, sh);
      end
    end
  end
end
