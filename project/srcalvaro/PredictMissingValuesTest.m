function X_pred = PredictMissingValues(X, nil)

k = 20; % number of clusters
d = 10; % dimensions to reduce to
iter = 5; % maximum number of iterations for training the model
seed = 14; % random seed

rows = @(x) size(x,1);
cols = @(x) size(x,2);

if isnan(nil)
  nils = isnan(X);
else
  nils = (X == nil);
  X(nils) = NaN;
end

[~, ~, ~, B] = ComputeBiases(X);
X = X - B;
X(nils) = 0;

fprintf('reducing dimensionality... ');
[~,X] = pcares(X,d);
fprintf('DONE\n');

for d = 2:10
  fprintf('--------------------------------\n');
  rand('seed', seed);
  fprintf('Fitting gaussian model... ');
  options = statset('MaxIter',iter);
  GMModel = gmdistribution.fit(X, k, 'Regularize',0.1, 'Options',options);
  fprintf('DONE\n');

  idx = cluster(GMModel, X);

  for clust = 1:k
    clusterMatches = (idx == clust);
    clusterMeans = repmat(mean(X(clusterMatches, :), 1), rows(X), 1);

    fprintf('cluster %d has %d matches\n', clust, sum(clusterMatches));

    updatePositions = nils & repmat(clusterMatches, 1, cols(X));
    X(updatePositions) = clusterMeans(updatePositions);
  end

  X_pred = X + B;
  fprintf('seed=%d K=%d BIC=%f RMSE=%f\n', seed, k, GMModel.BIC, RMSE(X_pred));
end
