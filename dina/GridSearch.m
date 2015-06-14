function [ ] = GridSearch(X_trn, X_tst, nil)

for pca = 25:5:50
            for p = 0:0.2:1
                X_pred = PredictMissingValues(X_trn, nil, pca, p);
                rmse = sqrt(mean((X_tst(X_tst ~= nil) - X_pred(X_tst ~= nil)).^2));
                disp([pca svd k p rmse])
            end
end


end

