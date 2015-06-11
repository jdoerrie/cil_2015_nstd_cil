function X_pred = PredictMissingValues(X, nil)

if isnan(nil)
  nils = isnan(X);
else
  nils = (X == nil);
  X(nils) = NaN;
end

fprintf('Learning biases... ');
[~, ~, ~, B] = LearnBiases(X);
fprintf('DONE\n');
X = X - B;
X(nils) = 0;

fprintf('Computing similarities... ');
[S, U] = ComputeSimilarity(X, nils);
fprintf('DONE\n');

X_pred = X + B;
