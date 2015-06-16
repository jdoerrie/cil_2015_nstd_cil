function [ X_pred ] = CallBaselines( X, nil )

X_pred = X;
nils = X_pred == nil;
X_pred(nils) = NaN;

X_avg = AverageBiases(X_pred);

X_pred = Baseline_Kmeans(X_avg, X_pred);

%X_pred = Baseline_SVD(X_avg);

%[something X_pred] = pcares(X_avg, 8);


end

