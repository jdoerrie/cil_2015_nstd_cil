function X_pred = PredictMissingValues(X, nil)
X_pred = X;
[u, it] = size(X_pred);
X_pred(X_pred == nil) = NaN;
XX = X_pred;
biases = nanmean(X_pred);
biases2 = nanmean(X_pred,2);
for i = 1:u
    for j = 1:it
        if (isnan(X_pred(i, j)))
            if (isnan(biases2(i)))
                X_pred(i, j) = biases(j);
            else
                X_pred(i, j) = biases(j)*0.75 + biases2(i)*0.25;
            end
        end
    end
end
%avg = mean(mean(X(X ~= nil)));  % BASELINE average over all values: 1.1211 MSE
%X_pred(X_pred == nil) = avg; % dummy prediction, 4 stars always

for asdf = 1:1
    k = 20;
    kk = 5;
    clusters = kmeans(X_pred, k, 'MaxIter', 10);
    clusters2 = kmeans(X_pred.', kk, 'MaxIter', 50);
    mean_vector = zeros(k, it);
    mean_vector2 = zeros(k, u);
    for i = 1:kk
        cluster2 = XX(:, clusters2 == i).';
        mean_vector2(i, :) = nanmean(cluster2);
    end
    for i = 1:k
        cluster = XX(clusters == i, :);
        mean_vector(i,:) = nanmean(cluster);
        [users, items] = size(cluster);
        for user = 1:u
            if (clusters(user) == i)
                for item = 1:it
                    if (X(user, item) == nil)
                        c = clusters2(item);
                        if (~isnan(mean_vector(i, item)) && ~isnan(mean_vector2(c, user)))
                            X_pred(user, item) = mean_vector(i, item)*0.7 + mean_vector2(c, user)*0.3;
                        elseif (~isnan(mean_vector(i, item)))
                            X_pred(user, item) = mean_vector(i, item);
                        elseif (~isnan(mean_vector2(c, user)))
                            X_pred(user, item) = mean_vector2(c, user);
                        end
                    end
                end
            end
        end
    end
end
    %}
    %{
    for user = 1:users
        this_cluster = cluster;
        for j = it:1
            if (X(user, j) == nil)
            this_cluster(:,j) = [];
            end
        end
        nn = knnsearch(this_cluster, this_cluster(user, :), 'K', 10);
        %nn2 = knnsearch(this_cluster.', this_cluster(user, :).', 'K', 10); 
        [x, y] = size(nn);
        for j = 1:it
            if (X(user, j) == nil)
                sums = zeros(x,1);
                count = 0;
                for n = 1:x
                    if (X(n, j) ~= nil)
                        sums(n) = X(n, j);
                        count = count + 1;
                    end
                end
                s = sum(sums)/count;
                if (count ~= 0)
                    X_pred(user, j) = s;
                else
                    X_pred(user, j) = mean_vector(i, j);
                end
            end
        end
    end
    %}
end