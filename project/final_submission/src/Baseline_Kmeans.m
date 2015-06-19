function [ X_pred ] = Baseline_Kmeans( X, k )

% Turn off warnings (as the max. iteration of k-means is set to 5, it
% usually does not converge and gives a warning).
warning('off','all');

% Initialize with compute biases.
[~, ~, ~, B] = ComputeBiases(X);
X_pred = B;

% Compute k clusters on the imputed data. Max. iteration is set to 5 due to
% the fact that more iterations barely improve the result.
clusters = kmeans(X_pred, k, 'MaxIter', 5);

% For each cluster, find the mean rating per item and assign it to every
% user in this cluster.
for i = 1:k
    cluster = X(clusters == i, :);
    cluster_item_mean = repmat(nanmean(cluster), size(cluster, 1), 1);
    X_pred(clusters == i, :) = cluster_item_mean;
end

% If no user of a cluster rated an item, its mean will be NaN. Those values
% are reset to the imputed values.
X_pred(isnan(X_pred)) = B(isnan(X_pred));

end

