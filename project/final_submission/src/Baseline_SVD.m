function [ X_pred ] = Baseline_SVD( X_pred, k, X_tst, nil )

%% Choose Initialization Technique

% Average: Overall mean
%mu = nanmean(X_pred(:));
%X_pred(isnan(X_pred)) = mu;

% Average Mean: average of user and item means
X_pred = AverageBiases(X_pred);

% Compute Biases or Learn Biases
%[~, ~, ~, B] = ComputeBiases(X_pred);
%[~, ~, ~, B] = LearnBiases(X_pred, 0.001, 0.02, X_tst, nil);
%X_pred(isnan(X_pred)) = B(isnan(X_pred));

%% SVD Part
% Perform SVD
[U, D, V] = svd(X_pred, 0);

% Prepare matrices, change dimensions due to k (cutoff)
sqrt_D = sqrt(D);
sqrt_D1 = sqrt_D(:, 1:k);
sqrt_D2 = sqrt_D(1:k, :);
U = U(:, 1:k);
V = V(:, 1:k);

% Multiply new, cutoff matrices to obtain prediction
U_prime = U * sqrt_D2;
V_prime = sqrt_D1 * V.';
X_pred = U_prime * V_prime;

end

