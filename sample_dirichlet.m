% p = sample_dirichlet(a[,N]) Samples from Dirichlet distribution
%
% x ~ Dir(a) where x, a \in R^D, with sum(xi) = 1 and xi >= 0 (i.e., x is
% in the D-dim unit simplex), means
%
%   p(x|a) \propto x1^(a1-1) * ... * xD^(aD-1).
%
% Particular cases:
% - ai = 1 -> uniform distribution on the simplex.
% - D = 2: Beta distribution.
%
% We use the method from p. 582 of:
%   A. Gelman, J. B. Carlin, H. S. Stern and D. B. Rubin:
%   "Bayesian Data Analysis" (2nd ed.), Chapman & Hall, 2004.
% Which is: draw y1,...,yD from independent gamma distributions with
% common scale and shape parameters a1,...,aD; for each i the
% sample is xi = yi/sum(yj).
%
% Examples:
%   p = sample_dirichlet([1 1 1],1000); -> uniform distribution on 3-simplex
%   p = sample_dirichlet([2 7 5],1000); -> concentrated on a side
% Plot with:
%   plot3(p(:,1),p(:,2),p(:,3),'r.'); xlabel('x'); ylabel('y'); zlabel('z');
%   set(gca,'DataAspectRatio',[1 1 1]); view(-45,-45);
%
% Note: requires 'gamrnd' (sampling from gamma distribution) from Matlab's
% Statistics Toolbox.
%
% In:
%   a: (1xD vector) parameter of the Dirichlet distribution (D dim).
%   N: number of samples to obtain. Default: 1.
% Out:
%   p: (NxD matrix) the N samples, one per row.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2004 by Miguel A. Carreira-Perpinan

function p = sample_dirichlet(a,N)

% ---------- Argument defaults ----------
if ~exist('N','var') | isempty(N) 
    N = 1; 
end;
% ---------- End of "argument defaults" ----------

D = length(a);
p = gamrnd(repmat(a,N,1),1,N,D);		% Samples from gamma
p = p ./ repmat(sum(p,2),1,D);			% Normalisation
end 