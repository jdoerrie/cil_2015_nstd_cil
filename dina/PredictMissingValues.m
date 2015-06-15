function X_pred = PredictMissingValues(X, nil)
warning('off','all');
%X_pred = X;
%[u, it] = size(X_pred);
%nils = X_pred == nil;
%X_pred(nils) = NaN;

%[X_pred, X_mean, X_std] = UserNormalization(X_pred);
X_pred = MyPCA(X, nil);
%[mu, b_u, b_i, B] = LearnBiases(X_pred);
%X_pred(nils) = B(nils);


%[something, X_pred] = pcares(zscore(X_pred), 30);
%X_pred = SKmeans(X_pred);
%[something, X_pred] = pcares(X_pred, 7);
%X_pred = X_pred .* X_std;
%X_pred = X_pred + X_mean;


end