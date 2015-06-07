function [P, Q, mu, bu, bi] = LearnVectors(X, K, gamma, lambda_1, lambda_2)
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

[mu, bu, bi] = LearnBiases(X, 0.1);
BaseLine = mu + bsxfun(@plus, bu, bi');
X = X - BaseLine;
% dimensions of X
[M, N] = size(X);

% I and J are the index vectors corresponding to non-entries entries in X,
% i.e. X(I(i), J(i)) is not NaN for all i
[I, J] = find(~isnan(X));

% numEntries is the total number of existing entries in the data matrix X
numEntries = length(I);

P = rand(M, K);
Q = rand(N, K);

rmse = RMSE(P * Q');
iter = 0;
while true
  iter = iter + 1;
  p = P;
  q = Q;
  for idx=1:numEntries
    u = I(idx);
    i = J(idx);

    r_ui = X(u,i);
    p_u = p(u,:);
    q_i = q(i,:);
    r_hat = p_u * q_i';

    e_ui = r_ui - r_hat; % - mu - bu(u) - bi(i)
    p(u,:) = p_u + gamma * (e_ui * q_i - lambda_1 * p_u);
    q(i,:) = q_i + gamma * (e_ui * p_u - lambda_1 * q_i);

    % compute current loss for given user/item combination
    % loss = X(u,i) - Q(u) - b_i(i)
    % update both bias vectors depending on the current loss, the regularize
    % parameter and the learning rate
    % b_u(u) = b_u(u) + (loss - lambda_1*b_u(u)) * u_lrate;
    % b_i(i) = b_i(i) + (loss - lambda_1*b_i(i)) * i_lrate;
  end

  fprintf('iter: %d\n', iter);
  X_pred = p*q' + BaseLine;
  err = RMSE(X_pred);
  fprintf('loss term: %f\n', err);
  eps = 1e-4;
  if rmse < err
    break
  end
  P = p;
  Q = q;
  if abs(rmse - err) < eps
    break;
  end
  rmse = err;
  % loss_mat = X - mu - bsxfun(@plus, bu, bi') - P*Q';
  % fprintf('sum loss: %f\n', nansum(loss_mat(:).^2));
  % reg_term = sum(P(:).^2) + sum(Q(:).^2) + sum(bu(:).^2) + sum(b(i).^2);
  % fprintf('reg term: %f, lambda_1: %f, total: %f\n', reg_term, lambda_1, reg_term*lambda_1);
end
end
