function [ X_pred ] = Baseline_SVD( X_pred )

[U, D, V] = svd(X_pred, 0);
sqrt_D = sqrt(D);
k = 8;
U = U(:, 1:k);
V = V(:, 1:k);
sqrt_D1 = sqrt_D(:, 1:k);
sqrt_D2 = sqrt_D(1:k, :);
U_prime = U * sqrt_D2;
V_prime = sqrt_D1 * V.';
X_pred = U_prime * V_prime;



end

