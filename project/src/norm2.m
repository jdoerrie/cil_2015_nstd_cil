function n = norm2(X)
  n = sum(X(:).^2, 'omitnan');
end
