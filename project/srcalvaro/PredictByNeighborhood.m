function X_pred = PredictByNeighborhood(X, P_pred, Q_pred, k, gamma)

if nargin < 4; k = 40; end
if nargin < 5; gamma = 0.1; end

[~,~,~,B] = ComputeBiases(X);

XMinusBias = X - B;
nils = isnan(XMinusBias);
XMinusBias(nils) = 0;
rated = ~nils;

S = ComputeSimilarity(XMinusBias);

X_pred = P_pred * Q_pred;

for i = 1:size(X,2)
  validSimilarities = bsxfun(@times, rated, S(i,:));
  [sortedSimilarities,I] = sort(validSimilarities, 'descend');
  I = I(1:k, :);
  sortedSimilarities = sortedSimilarities(1:k, :);

  X_pred(:,) - X(I(:,j), I(:,j));
  % take the proper elements from the matrix and perform the multiplicaton

  sum(sortedSimilarities);
end
