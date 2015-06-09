function X_pred = PredictMissingValues(X, nil, k, seed)

[num_users, num_items] = size(X);

if isnan(nil)
  nils = isnan(X);
else
  nils = (X == nil);
  X(nils) = NaN;
end

% means = repmat((nanmean(X)), num_users, 1);
%
% X(nils) = means(nils);
% X = X - means;

[mu, b_u, b_i, B] = ComputeBiases(X);
X = X - B;
X(nils) = 0;


fprintf('------------ training model ------------');
rand('seed', seed);
% the model with the lowest BIC is preferred
GMModel = fitgmdist(X, k, 'RegularizationValue',0.001);
fprintf('seed=%d BIC=%f', s, GMModel.BIC);

idx = cluster(GMModel, X);

for clust = 1:k
  clusterMatches = (idx == clust);
  clusterMeans = mean(X(clusterMatches, :), 1);
  fprintf('cluster %d has %d matches\n', clust, sum(clusterMatches));
  %X(clusterMatches, X(nils)) = clusterMeans;
  %nils(clusterMatches,:)
  nils2 = nils;
  nils2(~clusterMatches,:) = false;
  replicatedClusterMeans = repmat(clusterMeans, num_users, 1);
  X(nils2) = replicatedClusterMeans(nils2);
end

% X_pred = X + means;
X_pred = X + B;
