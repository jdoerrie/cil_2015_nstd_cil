function X_pred = DinaPCA(X, k)
if nargin < 2; k = 8; end

nils = isnan(X);
[X_pred, X_mean, X_std] = UserNormalization(X);

[~, ~, ~, B] = ComputeBiases(X_pred);
X_pred(nils) = B(nils);


[~, X_pred] = pcares(X_pred, k);

X_pred = X_pred .* X_std;
X_pred = X_pred + X_mean;
end