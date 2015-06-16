function [ X_pred ] = MyPCA( X, nil)

nils = X == nil;
X_nan = X;
X_nan(nils) = NaN;
notnils = ~isnan(X_nan);

mu = nanmean(X_nan(:));
X_mean = repmat(nanmean(X_nan), 10000, 1);
X_mean(isnan(X_mean)) = mu;

X_pred = X;
X_pred = X_pred - X_mean;


[users, items] = size(X_pred);


c = 10;
gamma = 0.01;
lambda = 0.25;
alpha = 5/8;


[U, D, V] = svd(X_pred, 0);
A = U(:, 1:c);
S = V(:, 1:c)';

Cost = zeros(users, items);
old_cost = 0;

for k = 1:23
    
    SpeedupA = zeros(users, c);
    SpeedupS = zeros(c, items);
    
    S_sq = bsxfun(@power, S, 2);
    A_sq = bsxfun(@power, A, 2);
    
    
    for i = 1:users
        SpeedupA(i, :) = sum(S_sq(:, notnils(i, :)), 2); 
    end

    for j = 1:items
        SpeedupS(:, j) = sum(A_sq(notnils(:, j), :)); 
    end

    A_new = bsxfun(@minus, A, bsxfun(@times, A, lambda));
    CostS = bsxfun(@times, Cost*S', gamma);
    SpeedupA = bsxfun(@power, SpeedupA, alpha);
    CostS(SpeedupA > 0) = CostS(SpeedupA > 0) ./ SpeedupA(SpeedupA > 0);
    A_new = bsxfun(@plus, A_new, CostS);
    
    S_new = bsxfun(@minus, S, bsxfun(@times, S, lambda));
    CostA = bsxfun(@times, Cost'*A, gamma)';
    SpeedupS = bsxfun(@power, SpeedupS, alpha);
    CostA(SpeedupS > 0) = CostA(SpeedupS > 0) ./ SpeedupS(SpeedupS > 0);
    S_new = bsxfun(@plus, S_new, CostA);
    
    X_bar = A_new*S_new;
    Cost(notnils) = bsxfun(@minus, X_pred(notnils),X_bar(notnils));
    sq_cost = bsxfun(@power, Cost, 2);
    cost_fun = sum(sq_cost(:));
    %vk = repmat(var(S, 0, 2), 1, items);
    %cost_fun = cost_fun/var(Cost(:))+sum(notnils(:)) * log(var(Cost(:)));% + sum(A(:).^2) + sum(sum(S.^2./vk + log(vk)))
    if (cost_fun < old_cost || k == 1) 
        gamma = gamma*1.1;
        A = A_new;
        S = S_new;
        old_cost = cost_fun;
        %X_t = bsxfun(@plus, A_new*S_new, X_mean);
        %rmse = sqrt(mean((X_tst(X_tst ~= nil) - X_t(X_tst ~= nil)).^2));
        %disp([k rmse]);
    else
        gamma = gamma/2;
    end 
end

X_pred = bsxfun(@plus, A*S, X_mean);


end
