function [A, W, v_coor] = gen_graph(n, r)
fail = 1;

while(fail == 1)
    v_coor = rand(n, 2);
    
    [X1, X2] = meshgrid(v_coor(:, 1), v_coor(:, 1));
    Xdiff = X1 - X2;
    [Y1, Y2] = meshgrid(v_coor(:, 2), v_coor(:, 2));
    Ydiff = Y1 - Y2;
    
    D_mat = sqrt(Xdiff.^2 + Ydiff.^2);
    
    A = (D_mat < r) - eye(n);
    
    % Test if A is  connected
    temp = zeros(n);
    Ak = A;
    k = 1;
    while(k < n)
        temp = temp + Ak;
        if( sum(sum(temp ~= 0)) == n^2 )
            break;
        end
        Ak = Ak * A;
        k = k + 1;
    end
    
    if(k < n)
        fail = 0;
    end
end


% Put weights according to MH rule
deg_V = sum(A, 2);
W = zeros(n);
for ii = 1:n
    crt_a = A(ii, :);
    j_idx = find(crt_a); % neighborhood of vertex ii
    W(ii, j_idx) = min(1./deg_V(ii), 1./deg_V(j_idx));
    W(ii, ii) = sum(max(0, 1./deg_V(ii) - 1./deg_V(j_idx)));
end
end