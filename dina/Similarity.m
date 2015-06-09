function [ D ] = Similarity( XI, XJ )

subtracted = XI - nanmean(XI, 2);

[users, items] = size(XJ);

D = zeros(users, 1);

for user = 1:users
    neighbor_subtracted = XJ(user, :) - nanmean(XJ(user, :), 2);
    D(user) = sum(bsxfun(@times, subtracted, neighbor_subtracted)) / sqrt(sum(bsxfun(@times, bsxfun(@times, subtracted, subtracted), bsxfun(@times, neighbor_subtracted, neighbor_subtracted))));
end


end

