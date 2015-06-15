function X_pred = PredictMissingValues(X, nil)

k = 25; % number of neighbours
betaParam = 500; % TODO fit parameter

if isnan(nil)
  nils = isnan(X);
else
  nils = (X == nil);
  X(nils) = NaN;
end

[~, ~, ~, B] = LearnBiases(X);
X = X - B;
X(nils) = 0;

[S, U] = ComputeSimilarity(X, nils); % TODO fit parameter lambda8

[~,I] = sort(S, 'descend'); % greater similarity -> more resemblance
I = I(1:k, :); % keep the top k neighbour positions

Abar = (X' * X) ./ U; % 5.24
Abar(isnan(Abar)) = 0;

avg = sum(sum(Abar - diag(diag(Abar)))) / (size(Abar, 1) * size(Abar, 2) - size(Abar, 1));
avgDiag = mean(diag(Abar));
Ahat = Abar .* U + betaParam * avg; % start 5.26
Ahat = Ahat - diag(repmat(betaParam * avg, size(Abar, 1), 1)); % substract betaParam*avg to diagonal
Ahat = Ahat + diag(repmat(betaParam * avgDiag, size(Abar, 1), 1)); % add proper value to diagonal
Ahat = Ahat ./ (U + betaParam); % finish 5.26

bhat = diag(Ahat); % 5.27 % FIXME

X_pred = zeros(size(X));
for j = 1:size(X, 2)
  A = Ahat(I(:,j), I(:,j)); % kxk
  b = bhat(I(:,j)); % kx1

  theta = linsolve(A, b);

  X_pred(:, j) = B(:, j) + X(:, I(:,j)) * theta; % 5.19
end
