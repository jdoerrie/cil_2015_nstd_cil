function X_pred = RidgeRegression(X_orig, X_preds, lambda)
  % Simple Implementation of Ridge Regression to blend several predictions
  % into a single one. X_orig is the original rating matrix, while X_preds
  % in a three dimensional array that contains the individual prediction
  % matrices for X in the last column, i.e. X_pred(:,:,i) contains the ith
  % prediction. Lambda is a regularizer constant.

  % set the default value of the regularizer if no value was specified
  if nargin < 3; lambda = 1e5; end

  % Obtain the dimensions of the input and find where values are present.
  [M, N] = size(X_orig);
  nPreds = size(X_preds, 3);
  NaNs = isnan(X_orig);
  nRatings = sum(~NaNs(:));

  % Construct the X matrix, where each column consists of the predictions
  % for the present values.
  X = zeros(nRatings, nPreds);
  for i=1:nPreds
    X_pred = X_preds(:,:,i);
    X(:,i) = X_pred(~NaNs);
  end

  % Y contains the "ground truth".
  Y = X_orig(~NaNs);

  % Closed Formula for Ridge Regression, finds W that minimizes
  % the difference between X*W and Y while penalizing large values in W
  W = ((X'*X + lambda*eye(nPreds)) \ X')*Y;

  % We reshape X_preds from size [M,N,P] to a 2D matrix of size [M*N,P] in
  % order to be able to take the product with W to obtain the final
  % weighted sum of individual predictions.
  X_preds = reshape(X_preds, M*N, nPreds);
  X_pred = X_preds * W;

  % X_pred currently is an array of size M*N, however we would like to have
  % a matrix of size [M,N], which is why another reshape is necessary. In
  % the end we clip the results to their valid range.
  X_pred = reshape(X_pred, M, N);
  X_pred = min(max(X_pred, 1), 5);
end
