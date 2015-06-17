function [trnSplits, tstSplits] = CrossValidationSplits(X, nSplits)
  % Given an input array X this method partitions X into a given number of
  % train and test splits.  The number of splits can be user defined and
  % defaults to 10.  It is guaranteed that the resulting train and test
  % splits are disjoint for every given fold.  However, it is not
  % guaranteed that every element of X appears at least once in the
  % tstSplits, this will only happen when nSplits divides the length of X.
  if nargin < 2; nSplits = 10; end

  nElements = length(X);
  tstSize = floor(nElements / nSplits);
  trnSize = nElements - tstSize;

  trnSplits = zeros(nSplits, trnSize);
  tstSplits = zeros(nSplits, tstSize);

  for i=1:nSplits
    test_idx = ((i-1)*tstSize + 1) : (i*tstSize);

    curr_tst = X(test_idx);
    curr_trn = setdiff(X, curr_tst);

    trnSplits(i,:) = curr_trn;
    tstSplits(i,:) = curr_tst;
  end
end
