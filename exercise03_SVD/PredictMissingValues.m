function X_pred = PredictMissingValues(X, nil)
% Predict missing entries in matrix X based on known entries. Missing
% values in X are denoted by the special constant value nil.

% your collaborative filtering code here!
X_pred = X;
for i = 1 : size(X,1)
    X_mean = 0;
    count = 0;
    for j = 1 : size(X,2)
        if (X(i, j)) ~= nil
            X_mean = X_mean + X_pred(i, j);
            count = count + 1;
        end 
    end
    X_mean = X_mean/count;
    for j = 1 : size(X,2)
        if (X(i, j)) == nil
            X_pred(i, j) = X_mean;
        end 
    end
end

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

for i = 1 : size(X,1)
    for j = 1 : size(X,2)
        if X(i, j) == nil
            X_pred(i, j) = res(i, j);
        end
    end
end
end