function I_rec = Decompress(I_comp)
% Your decompression code goes here!
% I_rec = double(I_comp.I) / 255;

[C, ~] = size(I_comp.U);
for c = 1:C
    I_comp.U{c} = double(I_comp.U{c}) / 127;
    I_comp.S{c} = double(I_comp.S{c});
    I_comp.V{c} = double(I_comp.V{c}) / 127;
end

[H, W] = size(I_comp.U{1} * I_comp.V{1});

I_rec = zeros(H,W,C);

for c = 1:C
  I_rec(:,:,c) = I_comp.U{c} * diag(I_comp.S{c}) * I_comp.V{c};
end

% d = double(I_comp.d);
% H = double(I_comp.H);
% W = double(I_comp.W);
% C = double(I_comp.C);

% % Hpad = H + rem(d - rem(H,d),d);
% % Wpad = W + rem(d - rem(W,d),d);

% I_rec = zeros(H,W,C);
% for c = 1:C
%  X_bar_rec = I_comp.Uk{c} * I_comp.Zk_bar{c};
%  X_rec = bsxfun(@plus, X_bar_rec, I_comp.mu{c});
%  I_rec(:,:,c) = col2im(X_rec, [d, d], [H W], 'sliding');
% %  I_rec(:,:,c) = X_rec(1:H, 1:W);
% end
% I_rec = I_comp.I; % this is just a stump to make the evaluation script run, replace it with your code!
