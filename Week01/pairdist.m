function D = pairdist(P, Q)
  % Fastest Locally
  % D = pdist2(P, Q);
  % Fastest on Judge
  C1 = bsxfun(@minus, P(:,1), Q(:,1)');
  C2 = bsxfun(@minus, P(:,2), Q(:,2)');
  D = sqrt(C1.^2 + C2.^2);
end
