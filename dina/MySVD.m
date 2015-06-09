function [ X_pred ] = MySVD( X_pred, XX )

[U, D, V] = svd(X_pred, 0);
sqrt_D = sqrt(D);
k = 6;
U = U(:, 1:k);
V = V(:, 1:k);
sqrt_D1 = sqrt_D(:, 1:k);
sqrt_D2 = sqrt_D(1:k, :);
U_prime = U * sqrt_D2;
V_prime = sqrt_D1 * V.';
res = U_prime * V_prime;

for i = 1 : size(XX,1)
    for j = 1 : size(XX,2)
        if isnan(XX(i, j))
            X_pred(i, j) = res(i, j);
        end
    end
end


end

