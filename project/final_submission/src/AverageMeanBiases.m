function [ X_pred ] = AverageMeanBiases( X, lambda )

% Weight of the user mean (and 1-lambda refers to the item mean)
if (nargin < 2) lambda  = 0.3; end

nils = isnan(X);

% Overall mean
mu = nanmean(X(:));

% User mean (set to mu if NaN), multiplied by its weight
user_means = repmat(nanmean(X, 2), 1, 1000);
user_means(isnan(user_means)) = mu;
user_means = bsxfun(@times, user_means, lambda);

% Item mean (set to mu if NaN), multiplied by its weight 1-lambda
item_means = repmat(nanmean(X, 1), 10000, 1);
item_means(isnan(item_means)) = mu;
item_means = bsxfun(@times, item_means, 1-lambda);

X_pred = X;

% Add means
X_pred(nils) = bsxfun(@plus, user_means(nils), item_means(nils));


end

