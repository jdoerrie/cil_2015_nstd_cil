function err = RMSE(X_pred)
global nil;
global X_tst;
X_pred = min(max(X_pred, 1), 5);
err = sqrt(mean((X_tst(X_tst ~= nil) - X_pred(X_tst ~= nil)).^2));  % error on known test values
end
