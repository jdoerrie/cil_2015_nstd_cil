function a = lassign(X, mu0, Sigma0, mu1, Sigma1)
  % mvnpdf expects X = NxD, mu = 1xD, Sigma = DxD
  Y0 = mvnpdf(X', mu0', Sigma0);
  Y1 = mvnpdf(X', mu1', Sigma1);

  a = ((Y0 < Y1) + 1)';
end
