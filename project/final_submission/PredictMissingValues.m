function X_pred = PredictMissingValues(X, nil)
  % Predict missing entries in matrix X based on known entries. Missing
  % values in X are denoted by the special constant value nil.

  if nargin < 2; nil = 0; end

  % replace nils with NaNs, this is because many of Matlab's builtin functions
  % denote missing values with NaN.
  if isnan(nil)
    nans = isnan(X);
  else
    nans = (X == nil);
    X(nans) = NaN;
  end


  % get the dimension of the input
  [M,N] = size(X);

  % setup the 3d predictoin matrix
  X_preds = zeros(M, N, 4);

  % add predictions of the different algorithms and the constant one term
  X_preds(:,:,1) = SVDpp(X, 32, 0.010, 0.02, 0.95);
  X_preds(:,:,2) = FactNgbrItem(X, 32, 0.005, 0.05, 0.825);
  X_preds(:,:,3) = FactNgbrUser(X, 64, 0.01, 0.05, 0.975);
  X_preds(:,:,4) = 1;

  % finally merge everything together to a sibgle prediction
  X_pred = BinnedRidgeRegression(X, X_preds, 1000, 100);
end
