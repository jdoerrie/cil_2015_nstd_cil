function [ X_pred ] = SKmeans( X_pred, XX, p )

[users, items] = size(XX);
[U, D, V] = svd(X_pred, 0);

a = 6;
D(a:end, a:end) = 0;

X_clust = U*D;
%X_clust = X_pred;
%[something, X_clust] = pcares(X_clust, 30);

k = 10;
clusters = kmeans(X_clust, k, 'MaxIter', 10);

%mean_vector = zeros(k, items)

for i = 1:k
    cluster = X_pred(clusters == i, :);
    %B = regSVD2(cluster);
    mean_vector = nanmean(cluster);
    mean_vector = repmat(mean_vector, size(cluster, 1), 1);
    X_pred(clusters == i, :) = X_pred(clusters == i, :) * (1-p) + mean_vector*p;
end


end

