function X_pred = PredictMissingValuesA(X, nil)
X_pred = X;
X_pred(X_pred == nil) = NaN;
biases = nanmean(X_pred);
for i = 1:10000
    for j = 1:1000
        if (isnan(X_pred(i, j)))
            X_pred(i, j) = biases(j);
        end
    end
end
%avg = mean(mean(X(X ~= nil)));  % BASELINE average over all values: 1.1211 MSE
%X_pred(X_pred == nil) = avg; % dummy prediction, 4 stars always

k = 20;
clusters = kmeans(X_pred, k, 'MaxIter', 10);
mean_vector = zeros(k, 1000);
for i = 1:k
    p = X_pred(clusters == i, :);
    mean_vector(i,:) = nanmean(p);
end
for i = 1:10000
    cluster = clusters(i);
    for j = 1:1000
        if X(i, j) == nil
            X_pred(i, j) = mean_vector(cluster, j);
        end
    end
end