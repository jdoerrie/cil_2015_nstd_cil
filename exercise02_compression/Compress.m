function I_comp = Compress(I, qual)
% Your compression code goes here
% I_comp.I = uint8(I * 255);
[H, W, C] = size(I);

I_comp.U = cell(C,1);
I_comp.S = cell(C,1);
I_comp.V = cell(C,1);

% qual = 0.0;
for c = 1:C
  [U,S,V] = svd(I(:,:,c));
  S = diag(S);

  k = find(cumsum(S) / sum(S) >= qual, 1);
  % fprintf('Considering the %d largest singular values\n', k);
  I_comp.U{c} = U(:,1:k);
  I_comp.S{c} = S(1:k);
  I_comp.V{c} = V(:,1:k)';
end
% d = 6;


for c = 1:C
  I_comp.U{c} = int8(I_comp.U{c} * 127);
  I_comp.S{c} = single(I_comp.S{c});
  I_comp.V{c} = int8(I_comp.V{c} * 127);
end

% X = extract(I, d);

% I_comp.Zmax = cell(C, 1);
% I_comp.Zk_bar = cell(C, 1);
% I_comp.mu = cell(C, 1);
% I_comp.Uk = cell(C, 1);

% for c = 1:C
%     [U, ~, mu, k] = PCAanalyse(X{c}, 50);
%     disp(sprintf('Using first %d principal components', k));
%     Uk = U(:,1:k);
%     X_bar = bsxfun(@minus, X{c}, mu);
%     Zk_bar = Uk' *  X_bar;
%     Zmax = max(abs(Zk_bar(:)));

%     I_comp.Zmax{c} = Zmax;
%     I_comp.Zk_bar{c} = single(Zk_bar);
%     I_comp.mu{c} = single(mu);
%     I_comp.Uk{c} = single(Uk);
% end
% %

% I_comp.d = uint8(d);
% I_comp.H = uint16(H);
% I_comp.W = uint16(W);
% I_comp.C = uint8(C);

%X_bar_rec = Uk*Zk_bar;
%X_rec = bsxfun(@plus, X_bar_rec, mu);
% I_comp.I = I; % this is just a stump to make the evaluation script run, replace it with your code!
