function [P_best, Q_best, mu, bu_best, bi_best] = LearnVectors(X, K, gamma, lambda_1, lambda_2)
% set default values of parameters
if nargin < 2
  K = 8;
end

if nargin < 3
  gamma = 0.001;
end

if nargin < 4
  lambda_1 = 0.02;
end

if nargin < 5
  lambda_2 = 0.05;
end

% dimensions of X
[N, M] = size(X);

% I and J are the index vectors corresponding to non-entries entries in X,
% i.e. X(I(i), J(i)) is not NaN for all i
[I, J] = find(~isnan(X));

% numEntries is the total number of existing entries in the data matrix X
numEntries = length(I);

P_best = rand(N, K);
Q_best = rand(M, K);

bu_best = zeros(N, 1);
bi_best = zeros(M, 1);
mu = reg_nanmean(X(:));

rmse = RMSE(P_best * Q_best');
iter = 0;
while 1
  iter = iter + 1;
  P = P_best;
  Q = Q_best;
  bu = bu_best;
  bi = bi_best;
  for idx=1:numEntries
    u = I(idx);
    i = J(idx);

    r_ui = X(u,i);
    p_u = P(u,:);
    q_i = Q(i,:);
    r_hat = bu(u) + bi(i) + p_u * q_i';

    e_ui = r_ui - r_hat; % - mu - bu(u) - bi(i)
    P(u,:) = p_u + gamma * (e_ui * q_i - lambda_1 * p_u);
    Q(i,:) = q_i + gamma * (e_ui * p_u - lambda_1 * q_i);
    bu(u) = bu(u) + gamma * (e_ui - lambda_2 * (bu(u) + bi(i) - mu));
    bi(i) = bi(i) + gamma * (e_ui - lambda_2 * (bu(u) + bi(i) - mu));

    % compute current loss for given user/item combination
    % loss = X(u,i) - Q(u) - b_i(i)
    % update both bias vectors depending on the current loss, the regularize
    % parameter and the learning rate
    % b_u(u) = b_u(u) + (loss - lambda_1*b_u(u)) * u_lrate;
    % b_i(i) = b_i(i) + (loss - lambda_1*b_i(i)) * i_lrate;
  end

  fprintf('iter: %d\n', iter);
  X_pred = P*Q' + bsxfun(@plus, bu, bi');
  err = RMSE(X_pred);
  fprintf('loss term: %f\n', err);
  eps = 1e-4;
  if rmse < err || abs(rmse - err) < eps
    break
  else
    P_best = P;
    Q_best = Q;
    bu_best = bu;
    bi_best = bi;
    rmse = err;
  % loss_mat = X - mu - bsxfun(@plus, bu, bi') - P*Q';
  % fprintf('sum loss: %f\n', nansum(loss_mat(:).^2));
  % reg_term = sum(P(:).^2) + sum(Q(:).^2) + sum(bu(:).^2) + sum(b(i).^2);
  % fprintf('reg term: %f, lambda_1: %f, total: %f\n', reg_term, lambda_1, reg_term*lambda_1);
end
end
