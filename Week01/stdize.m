function Y = stdize(X)
  Y = bsxfun(@minus, X, mean(X));
  Y = bsxfun(@rdivide, Y, std(X));
end
