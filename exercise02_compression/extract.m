function X = extract(I, d)
[H, W, C] = size(I);

Hpad = rem(d - rem(H,d),d);
Wpad = rem(d - rem(W,d),d);
% I = padarray(I, [Hpad Wpad], 'replicate', 'post');
H = H + Hpad;
W = W + Wpad;

% im2col does not support 3d matrices which is why this part needs to be
% done manually
X = cell(C, 1);
for c = 1:C
    X{c} = im2col(I(:,:,c), [d d], 'sliding');
end
