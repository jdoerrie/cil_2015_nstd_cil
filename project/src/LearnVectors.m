function [P, Q, mu, bu, bi] = LearnVectors(X, nil, K, gamma, lambda)
% set default values of parameters
if nargin < 2
  nil = 0;
end

if nargin < 3
  K = 8;
end

if nargin < 4
  gamma = 0.001;
end

if nargin < 5
  lambda = 0.5;
end

% set nil values to NaN
if ~isnan(nil)
  nils = (X == nil);
  X(nils) = NaN;
end

% dimensions of X
[N, M] = size(X);

% I and J are the index vectors corresponding to non-entries entries in X,
% i.e. X(I(i), J(i)) is not NaN for all i
[I, J] = find(~isnan(X));

% numEntries is the total number of existing entries in the data matrix X
numEntries = length(I);

P = rand(N, K);
Q = rand(M, K);

bu = zeros(N, 1);
bi = zeros(M, 1);
mu = reg_nanmean(X(:));

% number of iterations of SGD over whole dataset
numIter = 20;

for iter=1:numIter
  for idx=1:numEntries
    u = I(idx);
    i = J(idx);

    r_ui = X(u,i);
    p_u = P(u,:);
    q_i = Q(i,:);

    e_ui = r_ui - mu - bu(u) - bi(i) - q_i * p_u';
    P(u,:) = p_u + gamma * (e_ui * q_i - lambda * p_u);
    Q(i,:) = q_i + gamma * (e_ui * p_u - lambda * q_i);
    bu(u) = bu(u) + gamma * (e_ui - lambda * bu(u));
    bi(i) = bi(i) + gamma * (e_ui - lambda * bi(i));

    % compute current loss for given user/item combination
    % loss = X(u,i) - Q(u) - b_i(i)
    % update both bias vectors depending on the current loss, the regularize
    % parameter and the learning rate
    % b_u(u) = b_u(u) + (loss - lambda*b_u(u)) * u_lrate;
    % b_i(i) = b_i(i) + (loss - lambda*b_i(i)) * i_lrate;
  end

  fprintf('iter: %d\n', iter);
  loss_mat = X - mu - bsxfun(@plus, bu, bi') - P*Q';
  fprintf('sum loss: %f\n', nansum(loss_mat(:).^2));
  reg_term = sum(P(:).^2) + sum(Q(:).^2) + sum(bu(:).^2) + sum(b(i).^2);
  fprintf('reg term: %f, lambda: %f, total: %f\n', reg_term, lambda, reg_term*lambda);
end
end
