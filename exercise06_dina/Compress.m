function I_comp = Compress(I)
I_comp.clusters = 5;
[H, W, C] = size(I);
%cols = im2col(I, [3 1]);
if (C == 3) 
    cols = reshape(I, [], 3, 1).';
    %{
    [z1, U1, loglike1] = gmm(I(:,:,1), I_comp.clusters);
    I_comp.U1 = U1;
    I_comp.z1 = z1;
    [z2, U2, loglike2] = gmm(I(:,:,2), I_comp.clusters);
    I_comp.U2 = U2;
    I_comp.z2 = z2;
    [z3, U3, loglike3] = gmm(I(:,:,3), I_comp.clusters);
    I_comp.U3 = U3;
    I_comp.z3 = z3;
    %}
    [z, U, loglike] = gmm(cols, I_comp.clusters);
    I_comp.z = z;
    I_comp.U = U;
else 
    cols = I;
    [z, U, loglike] = gmm(cols, I_comp.clusters);
    I_comp.U = U;
    I_comp.z = z;
end

I_comp.H = H;
I_comp.W = W;
I_comp.C = C;

%I_comp.I = I; % this is just a stump to make the evaluation script run, replace it with your code!
