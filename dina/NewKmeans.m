function [ X_pred ] = NewKmeans( X_pred, XX )
X_init = X_pred;
k = 10;
kk = 20;
[u, it] = size(XX);
[clusters something anything D] = kmeans(X_pred, k, 'MaxIter', 10);
%{
D = pdist(X_init);
D = squareform(D);
disp('done');
%}
mean_vector = zeros(k, it);
for i = 1:k
    cluster = XX(clusters == i, :);
    [users, items] = size(cluster);
    mean_vector(i, :) = nanmean(cluster, 1);
%{
    for user = 1:u
        if (clusters(user) == i)
            for item =  1:it
                if (isnan(XX(user, item)))
                    rating = zeros(1,5);
                    for neighbor = 1:u
                        if (clusters(neighbor) == i) && ~isnan(XX(neighbor, item))
                            rating(XX(neighbor, item)) = rating(XX(neighbor, item)) + 1/(D(user, neighbor))^2;
                        end
                    end
                    [x, rating] = max(rating);
                    X_pred(user, item) = rating;
                end
            end
        end
    end
    %}
    %{
    for item = 1:it
        rating = zeros(1,5);
        for a = 1:5
            for user = 1:users
                if (~isnan(cluster(user, item)))
                    rating(cluster(user, item)) = rating(cluster(user, item)) + 1;
                end
            end
        end
        [x, rating] = max(rating);
        for user = 1:u
            if (clusters(user) == i)
                if (isnan(XX(user, item)) && ~isnan(means(item)))
                    X_pred(user, item) = means(item);
                end
            end
        end
    end  
  %}
    

    
    %{
    item_clusters = kmeans(cluster.', kk, 'MaxIter', 5);
    for j = 1:kk
        item_cluster = XX(clusters == i, item_clusters == j);
        means = nanmean(nanmean(item_cluster, 1));
        [users, items] = size(item_cluster);
        for user = 1:u
            if (clusters(user) == i)
                for item = 1:it
                    if (item_clusters(item) == j)
                        if (isnan(XX(user, item)) && ~isnan(means))
                            %X_pred(user, item) = X_pred(user, item)*0.5 + means*0.5;
                            X_pred(user, item) = means;
                        end
                    end
                end
            end
        end
    end
    %}
end

for user = 1:u
    for item = 1:it
        if isnan(XX(user, item))
            rating = 0;
            ssum = 0;
            kkk = 0;
            %ssum = sum(D(user, ~isnan(D(user, :))));
            for c = 1:k
                if ~isnan(mean_vector(c, item))
                    ssum = ssum + D(user, c);
                    kkk = kkk + 1;
                end
            end
            for c = 1:k
                if ~isnan(mean_vector(c, item))
                    weight =  (1- (D(user, c)/ssum))/(kkk-1);
                    rating = rating + mean_vector(c, item) * weight;
                end
            end
            X_pred(user, item) = rating;
        end
    end
end

end

