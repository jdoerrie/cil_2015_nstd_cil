function [ X_pred ] = MyBiases( X_pred )

biases = nanmean(X_pred);
biases2 = nanmean(X_pred,2);
for i = 1:size(X_pred, 1)
    for j = 1:size(X_pred, 2)
        if (isnan(X_pred(i, j)))
            if (isnan(biases2(i)))
                X_pred(i, j) = biases(j);
            else
                X_pred(i, j) = biases(j)*0.75 + biases2(i)*0.25;
            end
        end
    end
end

end

