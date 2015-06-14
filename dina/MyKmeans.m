function [ X_pred ] = MyKmeans( X_pred, XX, u, it )

k = 10;
kk = 5;
clusters = kmeans(X_pred, k, 'MaxIter', 5);
clusters2 = kmeans(X_pred.', kk, 'MaxIter', 5);
mean_vector = zeros(k, it);
mean_vector2 = zeros(k, u);
for i = 1:kk
    cluster2 = X_pred(:, clusters2 == i).';
    mean_vector2(i, :) = nanmean(cluster2);
end
for i = 1:k
    cluster = X_pred(clusters == i, :);
    mean_vector(i,:) = nanmean(cluster);
    [users, items] = size(cluster);
    for user = 1:u
        if (clusters(user) == i)
            for item = 1:it
                if (isnan(XX(user, item)))
                    c = clusters2(item);
                    if (~isnan(mean_vector(i, item)) && ~isnan(mean_vector2(c, user)))
                        X_pred(user, item) = X_pred(user, item)*0 + (mean_vector(i, item)*0.7 + mean_vector2(c, user)*0.3)*1;
                    elseif (~isnan(mean_vector(i, item)))
                        X_pred(user, item) = X_pred(user, item)*0.5 + mean_vector(i, item)*0.5;
                    elseif (~isnan(mean_vector2(c, user)))
                        X_pred(user, item) = X_pred(user, item)*0.5 + mean_vector2(c, user)*0.5;
                    end
                end
            end
        end
    end
end


end

