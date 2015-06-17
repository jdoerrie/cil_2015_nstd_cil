function [mu, b_u, b_i, B] = ComputeBiases(X, l1, l2)
  if (nargin < 2) l1 =  0; end % regularizer term for item biases
  if (nargin < 3) l2 = 12; end % regularizer term for user biases

  mu = nanmean(X(:));
  X = X - mu;

  b_i = reg_nanmean(X, 1, l1);
  b_i(isnan(b_i)) = 0;
  X = bsxfun(@minus, X, b_i);
  b_i = b_i';
  b_u = reg_nanmean(X, 2, l2);
  b_u(isnan(b_u)) = 0;

  B = mu + bsxfun(@plus, b_u, b_i');
  % fprintf('CompBias: l1 = %02d, l2 = %02d, RMSE: %f\n', l1, l2, RMSE(B));
end
