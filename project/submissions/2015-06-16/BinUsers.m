function bins = BinUsers(X, nBins)
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
