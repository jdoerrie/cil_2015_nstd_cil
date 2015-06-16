function [X_pred, allW] = BinnedRidgeRegression(X_orig, X_preds, nBins, lambda)
  % Simple Implementation of Ridge Regression to blend several predictions
  % into a single one. X_orig is the original rating matrix, while X_preds
  % in a three dimensional array that contains the individual prediction
  % matrices for X in the last column, i.e. X_pred(:,:,i) contains the ith
  % prediction. Lambda is a regularizer constant.

  % set the default value of the regularizer if no value was specified
  if nargin < 4; lambda = 1e5; end

  X_pred = zeros(size(X_orig));
  bins = BinUsers(X_orig, nBins);
  nPreds = size(X_preds, 3);

  allW = zeros(nPreds, nBins);
  for i=1:nBins
    idx = bins{i};
    [X_pred(idx,:), currW] = RidgeRegression(...
      X_orig(idx,:), X_preds(idx,:,:), lambda);
    allW(:,i) = currW;
  end
end
