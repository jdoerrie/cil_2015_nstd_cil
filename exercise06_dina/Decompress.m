function I_rec = Decompress(I_comp)
    if (I_comp.C == 3)
        %{
        I_rec = zeros(I_comp.H, I_comp.W, 3);
        one = recompute(I_comp.z1, I_comp.U1, I_comp.clusters);
        I_rec(:,:,1) = I_comp.U1 * one;
        two = recompute(I_comp.z2, I_comp.U2, I_comp.clusters);
        I_rec(:,:,2) = I_comp.U2 * two;
        three = recompute(I_comp.z3, I_comp.U3, I_comp.clusters);
        I_rec(:,:,3) = I_comp.U3 * three;
        %}
        rec = I_comp.U * recompute(I_comp.z, I_comp.U, I_comp.clusters);
        I_rec = reshape(rec.', I_comp.H, [], 3);
    else
        I_rec = I_comp.U * recompute(I_comp.z, I_comp.U, I_comp.clusters);
    end
    %I_rec = rand(size(I_rec));
    
end

function Z = recompute(z, U, clusters)
    s = size(z);
    Z = zeros(clusters, s(2));

    for i = 1 : s(2)
        Z(z(i), i) = 1;
    end
    
end

%I_rec = col2im(I_comp.U * I_comp.Z, [3 1], I_comp.size, 'sliding');

%I_rec = I_comp.I; % this is just a stump to make the evaluation script run, replace it with your code!