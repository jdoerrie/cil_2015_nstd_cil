function X_pred = PredictMissingValues(X, nil)

k = 25; % number of neighbours
betaParam = 500;

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

[S,I] = sort(S, 'descend'); % greater similarity -> more resemblance
I = I(1:k, :); % keep the top k neighbour positions

for j = 1:size(X, 2)
  Ujl = U(I(:,j), repmat(j, k, 1));
  Abar = X(:, I(:,j));
  Abar = (Abar' * Abar) ./ Ujl; % 5.24

  Uij = U(I(:,j), j);
  bbar = (X(:, j)' * X(:, I(:,j)))' ./ Uij; % 5.25

  A = (Ujl .* Abar + betaParam * avg) ./ (Ujl + betaParam); % 5.26
  b = (Uij ./ bbar + betaParam * avg) ./ (Uij + betaParam); % 5.27
end

X_pred = X + B;
