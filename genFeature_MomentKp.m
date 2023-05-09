function [ c,lambda,lambda_abs ] = genFeature_MomentKp(xdat, N, feature_set)
% xdat: input raw data
% c: vector coefficent defining companion matrix
% lambda: Koopman eigenvalues (empirical Ritz value)
% V_norm: norm of Koopman eigenvector (empirical Ritz vector)
    lambda = [];
    V_norm = [];
    [dim_s, dim_t] = size(xdat);
    if strcmp(feature_set, 'Arnoldi')
    %% Arnoldi-type algorithm
        % get constants c_j
        P = xdat(:,1:N-1);
        b = P' * xdat(:,end);
        A = P' * P;
        if rank(A) < length(b)
            c = pinv(A)*b;
        else
            c = A \ b;
        end
        N = length(c) + 1;
        % define the companion matrix C
        C = [[zeros(1,N-2);eye(N-2)], c];
        % Koopman eigenvalues, empirical Ritz values
        lambda = eig(C);
        lambda_abs = abs(lambda);
        % define the Vandermonde matrix T
        T = fliplr(vander(lambda));
        % Koopman modes, empirical Ritz vector
        V = P * inv(T);
        V_mod = abs(V);
        V_ang = angle(V);
        V_norm = diag(sqrt(V_mod'*V_mod)); % norm of each column of V_mod
        % Matrix approximation of Koopman operator U
        U = real(V * diag(lambda) * pinv(V));
    elseif strcmp(feature_set, 'Prony')
        %% Vector Prony Analysis
        N = N;
        H = zeros(dim_s*N, N); % Hankel Matrix
        for k = 1:(N)
            H((1+(k-1)*dim_s):(k*dim_s), :) = xdat(:, k:N-1+k);
        end
        b = xdat(:,(N+1):2*N); b = b(:);
        % p = pinv(H) * b; % if H is full rank, p=(H'*H) \ H'*b;
        if rank(H) ~= size(H,2)
            c = pinv(H) * b;
        else
            c = (H' * H) \ H' * b;
        end
        % define the companion matrix C
        C = [[zeros(1,N-1);eye(N-1)], -c];
        % Koopman eigenvalues, empirical Ritz values
        lambda = eig(C);
        lambda_abs = abs(lambda);
        % define the Vandermonde matrix T
        T = fliplr(vander(lambda));
        % Koopman modes, empirical Ritz vector
        V = xdat(:,1:N) / T;
        V_mod = abs(V);
        V_ang = angle(V);
        V_norm = diag(sqrt(V_mod'*V_mod)); % norm of each column of V_mod
    else
        disp('error');
    end
    [lambda_abs, lambda_abs_order] = sort(lambda_abs,'descend');
    lambda = lambda(lambda_abs_order);
end
