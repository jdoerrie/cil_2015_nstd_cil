function X_pred = PredictMissingValues(X, nil)

k = 25; % number of neighbours
betaParam = 500; % TODO fit parameter

if isnan(nil)
  nils = isnan(X);
else
  nils = (X == nil);
  X(nils) = NaN;
end

[X_pred, P_pred, Q_pred] = SVDpp(X);
X_pred = PredictByNeighborhood(X, P_pred, Q_pred);
