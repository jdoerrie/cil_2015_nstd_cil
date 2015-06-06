function X_pred = PredictMissingValues(X, nil)
X_pred = X;
[u, it] = size(X_pred);
X_pred(X_pred == nil) = NaN;
biases = nanmean(X_pred);
for i = 1:u
    for j = 1:it
        if (isnan(X_pred(i, j)))
            X_pred(i, j) = biases(j);
        end
    end
end
%avg = mean(mean(X(X ~= nil)));  % BASELINE average over all values: 1.1211 MSE
%X_pred(X_pred == nil) = avg; % dummy prediction, 4 stars always

k = 20;
clusters = kmeans(X_pred, k, 'MaxIter', 10);
mean_vector = zeros(k, it);
for i = 1:k
    p = X_pred(clusters == i, :);
    mean_vector(i,:) = nanmean(p);
    [users, items] = size(p);
    for user = 1:users
        cluster = p;
        for j = it:1
            if (X(user, j) == nil)
            cluster(:,j) = [];
            end
        end
        nn = knnsearch(cluster, cluster(user, :), 'K', 10);
        [x, y] = size(nn);
        for j = 1:it
            if (X(user, j) == nil)
                sum = 0;
                count = 0;
                for n = 1:x
                    if (X(n, j) ~= nil)
                        sum = sum + X(n, j);
                        count = count + 1;
                    end
                end
                if (sum ~= 0)
                    X_pred(user, j) = sum/count;
                else
                    X_pred(user, j) = mean_vector(i, j);
                end
            end
        end
    end
end