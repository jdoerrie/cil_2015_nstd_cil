function bins = BinUsers(X, nBins)
  % This method bins the users of X into nBins depending on how many
  % ratings they issued in X. X is asusmed to have NaNs for missing values.
  % The bins are returned as a cell array, because they can not be of equal
  % size when nBins does not divide the total number of users, hence a
  % fixed size array is not possible.

  nRatings = sum(~isnan(X), 2);
  totalRatings = sum(nRatings);

  [V, I] = sort(nRatings);
  C = cumsum(V) * nBins;
  bins = cell(nBins, 1);
  prevIdx = 1;
  for i=1:nBins
    idx = find(C <= totalRatings * i, 1, 'last');
    bins{i} = I(prevIdx:idx);
    prevIdx = idx+1;
  end
end
