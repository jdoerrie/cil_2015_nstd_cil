function X_pred = PredictMissingValues(X, nil)
  % Predict missing entries in matrix X based on known entries. Missing
  % values in X are denoted by the special constant value nil.

  % your collaborative filtering code here!

  if nargin < 2
    nil = 0;
  end

  % replace nils with NaNs
  if isnan(nil)
    nans = isnan(X);
  else
    nans = (X == nil);
    X(nans) = NaN;
  end
  
  
%   X_pred = zeros(size(X));
%   X_pred = X_pred + 0.40 * SVDpp(X, 64, 0.02, 0.02, 0.775);
%   X_pred = X_pred + 0.55 * FactNgbr(X, 64, 0.01, 0.1, 0.775);
%   X_pred = X_pred + 0.05 * FactNgbrUser(X, 64, 0.01, 0.1, 0.975);
  
%        err = RMSE(X_pred);
%        fprintf('Ridge Regression: Lambda = %d, RMSE = %f\n', lam, err); 
%    end
  
%   shrink = [0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 1.00];
%   for gam=range
%     for lam=range
%       for sh=shrink
%         X_pred = SVDpp(X, 64, gam, lam, sh);
%       end
%     end
%   end
end
