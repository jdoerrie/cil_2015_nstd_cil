function X_pred = PredictMissingValues(X, nil)

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


for seed = 1:10
  for k = 5
    fprintf('------------ training model ------------\n');
    rand('seed', seed);
    % the model with the lowest BIC is preferred
    GMModel = fitgmdist(X, k, 'RegularizationValue',0.001);

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
    X_pred = X + B;
    fprintf('seed=%d K=%d BIC=%f RMSE=%f\n', seed, k, GMModel.BIC, RMSE(X_pred));
  end
end
