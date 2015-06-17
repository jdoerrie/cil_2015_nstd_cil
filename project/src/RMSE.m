function err = RMSE(X_pred, X_test, nul)
  global nil;
  global X_tst;

  if nargin < 2; X_test = X_tst; end
  if nargin < 3; nul = nil; end
  
  X_test(isnan(X_test)) = nul;
  X_pred = min(max(X_pred, 1), 5);
  err = sqrt(mean((X_test(X_test ~= nul) - X_pred(X_test ~= nul)).^2));  % error on known test values
end
