function [ X_pred ] = SVDKmeans( X_gt, X_svd )

[users, items] = size(X_svd);

X_left = bsxfun(@minus, X_gt, X_svd);
X_init = X_left;
X_left(isnan(X_left)) = 0;

k = 10;
clusters = kmeans(X_left, k, 'MaxIter', 5);
mean_vector = zeros(k, items);

for i = 1:k
    cluster = X_init(clusters == i, :);
    mean_vector(i, :) = nanmean(cluster);
end

for user = 1:users
    c = clusters(user);
    for item = 1:items
        if (isnan(X_gt(user, item)) && ~isnan(mean_vector(c, item)))
            X_left(user, item) = mean_vector(c, item);
        end
    end
end

X_pred = X_gt;
X_sum = bsxfun(@plus, X_svd, X_left);
X_pred(isnan(X_pred)) = X_sum(isnan(X_pred));
end

