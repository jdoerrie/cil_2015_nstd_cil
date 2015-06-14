function [ X, X_mean, X_std ] = UserNormalization( X, XX )

[users, items] = size(X);
q = 5;
mu = nanmean(nanmean(X));
sigma_sq = nanmean(nanvar(X, 0, 2));


rating_counts = sum(~isnan(X), 2);
few_ratings = rating_counts < 50;
normalization = rating_counts(few_ratings, :) + q;

X_mean = nanmean(X, 2);
X_mean(few_ratings, :) = (sum(X(few_ratings, :), 2) + q*mu) ./ normalization;
X_mean = repmat(X_mean, 1, items);

X_std = nanstd(X, 0, 2);
X_few_sq_diffs = sum((X(few_ratings, :) - mu) .^ 2, 2);
X_std(few_ratings, :) = sqrt((X_few_sq_diffs + q*sigma_sq) ./ normalization);
X_std = repmat(X_std, 1, items);

X_mean(isnan(X_mean)) = mu;
X_std(isnan(X_std)) = sqrt(sigma_sq);
X_std(X_std == 0) = sqrt(sigma_sq);

X(isnan(X)) = 0;

X_diff = X - X_mean;
X = X_diff ./ X_std;

end

