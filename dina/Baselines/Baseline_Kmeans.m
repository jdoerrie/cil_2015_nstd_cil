function [ X_pred ] = Baseline_Kmeans( X_avg, X_nan )

k = 10;
X_pred = X_avg;
clusters = kmeans(X_avg, k, 'MaxIter', 5);

for i = 1:k
    cluster = X_nan(clusters == i, :);
    cluster_item_mean = repmat(nanmean(cluster), size(cluster, 1), 1);
    X_pred(clusters == i, :) = cluster_item_mean;
end

X_pred(isnan(X_pred)) = X_avg(isnan(X_pred));

end

