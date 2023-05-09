function [Mt2] = CalTempLegMoments(t, X, m)
%CalTempMoments calculate the temporal moments (Smallwood) from the given
%time series data
%   Input: 
%           X: arranged in feature dimension d x time dimension 
%           m: number of moments taken
%   Output:
%           Mt: temporal moments

Mt2 = zeros(size(X,1), m);

t_new = linspace(0,1,length(t));
a2 = trapz(t_new,t_new.*(X.^2),2)./(trapz(t_new,X.^2,2));
for i = 1:m
    for d = 1:size(X,1)
        Mt2(d,i) = trapz(t,sqrt((2*i-1)./2).*legpoly((i-1),(t_new-a2(d))).*X(d,:).^2,2);
    end
end

end