function X_pred = PredictMissingValues(X, nil)

k = 5;
seed = 6;

[num_users, num_items] = size(X);

if isnan(nil)
  nils = isnan(X);
else
  nils = (X == nil);
  X(nils) = NaN;
end

[mu, b_u, b_i, B] = ComputeBiases(X);
X = X - B;
X(nils) = 0;

rand('seed', seed);
% the model with the lowest BIC is preferred
GMModel = gmdistribution.fit(X, k, 'Regularize',0.001);

idx = cluster(GMModel, X);

for clust = 1:k
  clusterMatches = (idx == clust);
  clusterMeans = mean(X(clusterMatches, :), 1);

  nils2 = nils;
  nils2(~clusterMatches,:) = false;
  replicatedClusterMeans = repmat(clusterMeans, num_users, 1);
  X(nils2) = replicatedClusterMeans(nils2);
end

X_pred = X + B;
