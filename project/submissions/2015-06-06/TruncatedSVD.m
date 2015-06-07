function [X_bar, P, Q] = TruncatedSVD(X, K)
  [U, S, V] = svd(X, 0);
  U = U(:,1:K);
  S = S(1:K,1:K);
  V = V(:,1:K)';
  X_bar = U * S * V;
  P = U * sqrt(S);
  Q = sqrt(S) * V;
end
