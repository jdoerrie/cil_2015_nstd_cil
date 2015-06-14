function X_pred = PredictMissingValues(X, nil, p)
warning('off','all');
X_pred = X;
[u, it] = size(X_pred);
nils = X_pred == nil;
X_pred(nils) = NaN;

[X_pred, X_mean, X_std] = UserNormalization(X_pred);

[mu, b_u, b_i, B] = LearnBiases(X_pred);
X_pred(nils) = B(nils);


[something, X_pred] = pcares(X_pred, 8);

X_pred = X_pred .* X_std;
X_pred = X_pred + X_mean;


end