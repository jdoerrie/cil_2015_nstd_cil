function [ X ] = AverageBiases( X )

nils = isnan(X);

mu = nanmean(X(:));

user_means = repmat(nanmean(X, 2), 1, 1000);
user_means(isnan(user_means)) = mu;

item_means = repmat(nanmean(X, 1), 10000, 1);
item_means(isnan(item_means)) = mu;

X(nils) = (user_means(nils) + item_means(nils))/2;


end

