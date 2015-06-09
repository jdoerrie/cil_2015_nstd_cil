function X_pred = PredictMissingValues(X, nil)
warning('off','all');
X_pred = X;
[u, it] = size(X_pred);
nils = X_pred == nil;
X_pred(nils) = NaN;
XX = X_pred;
[mu, b_u, b_i, B] = ComputeBiases(X_pred);
X_pred(nils) = B(nils);
%X_pred = MyBiases(X_pred);


%{
for i = 1:u
    X_i = X_pred;
    for j = 1:it
        if (isnan(XX(i, j)))
            X_i(:, j) = [];
        else
            [users, items] = size(X_i);
            for k = 1:users
                if (k ~= i && isnan(XX(k, j)))
                    X_i(k, :) = [];
                end
            end
        end
    end
    knnsearch(X_i, X_i(i,:), 'K', 20);
end
%}
%avg = mean(mean(X(X ~= nil)));  % BASELINE average over all values: 1.1211 MSE
%X_pred(X_pred == nil) = avg; % dummy prediction, 4 stars always

for asdf = 1:3
    %X_pred = MySVD(X_pred, XX);
    %rmse = sqrt(mean((X_tst(X_tst ~= nil) - X_pred(X_tst ~= nil)).^2))
end

%save('asdf.mat', 'X_pred');

load('asdf.mat');
[something, X_pred] = pcares(X_pred, 30);



for asdf = 1:1
    X_pred = MyKmeans(X_pred, XX, u, it);
   % X_pred = NewKmeans(X_pred, XX);
    %rmse = sqrt(mean((X_tst(X_tst ~= nil) - X_pred(X_tst ~= nil)).^2));
    %disp([num2str(asdf) ' RMSE: ' num2str(rmse)]);
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