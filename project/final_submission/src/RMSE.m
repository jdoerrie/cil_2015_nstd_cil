function err = RMSE(X_pred, X_tst, nil)
  % Implementation of root mean sqquared error which is used to evaluate
  % the current predictions.

  % set remaning NaNs in the data to nil
  X_tst(isnan(X_tst)) = nil;

  % clamp X_pred to the valid range
  X_pred = min(max(X_pred, 1), 5);

  % error on known test values
  err = sqrt(mean((X_tst(X_tst ~= nil) - X_pred(X_tst ~= nil)).^2));
end
