function pval = legpoly (Q,x)
    [m,l] = size(x);
    if m < l;x = x'; 
    elseif m > 1 && l > 1
        x = x(:);
    end
    
    n = size(x,1);  % x: size n*1
    z = zeros(Q+1,n);
    if Q == 0
        z(1,:) = ones(1,n);
    elseif Q == 1
        z(2,:) = x;
    else
        z(1,:) = ones(1,n);
        z(2,:) = x;        
        for k = 3:Q+1
           k_1 = k-1;
           z(k,:) = ((2*k_1-1)/k_1)*x'.*z(k-1,:)-((k_1-1)/k_1)*z(k-2,:);
        end
%     pval = sqrt(2*(Q+1)/2) * z(end,:);
    end
    pval =  z(end,:);
%     pval = reshape(pval,[m,l]);
end