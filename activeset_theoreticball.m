function info = theoretic_polytope_baseline()
% Weighted-L1 (diamond) region in (δx0, δφ) from exponential-decay bound
% Model: x_{t+1} = x_t + u_t, box bounds on u_0..u_{T-1} and x_1..x_{T-1},
% terminal equality x_T = φ, stage cost (x-ξ)'Q(x-ξ) + u'R u

%% Problem size, parameters
T     = 10;
x0    = 0.0;           % baseline
phi   = 0.0;
xmax  = 5.0;
umax  = 1.0;

% costs (scalar or length-T)
Q_in  = 1.0;
R_in  = 1.0;

% reference: user block repeated
xi_block = [1, -1, 2, 0.7];
xi_vec   = repmat(xi_block(:), ceil(T/numel(xi_block)), 1);
xi_vec   = xi_vec(1:T);

% normalize
Q_vec = isscalar(Q_in)*repmat(Q_in,T,1) + (~isscalar(Q_in))*Q_in(:);
R_vec = isscalar(R_in)*repmat(R_in,T,1) + (~isscalar(R_in))*R_in(:);

%% Index helpers, z = [x0..xT, u0..u_{T-1}]
nx = T+1; nu = T; nz = nx+nu;
xidx = @(t) (t+1);          % t=0..T
uidx = @(t) (nx + t + 1);   % t=0..T-1

%% YALMIP model at baseline
x = sdpvar(nx,1); u = sdpvar(nu,1);
obj = sum(Q_vec .* (x(1:T)-xi_vec).^2) + sum(R_vec .* (u.^2));
F = [];
dyn = cell(T,1);
for t=1:T, c=(x(t+1)==x(t)+u(t)); F=[F,c]; dyn{t}=c; end
eq_x0 = (x(1)==x0); eq_xT=(x(T+1)==phi); F=[F,eq_x0,eq_xT];

ub_u=cell(T,1); lb_u=cell(T,1);
for t=1:T,  c1=( u(t) <= umax); c2=(-u(t) <= umax);
    F=[F,c1,c2]; ub_u{t}=c1; lb_u{t}=c2; end
ub_x=cell(T-1,1); lb_x=cell(T-1,1);
for t=2:T, c1=( x(t) <= xmax); c2=(-x(t) <= xmax);
    F=[F,c1,c2]; ub_x{t-1}=c1; lb_x{t-1}=c2; end

opts = sdpsettings('solver','gurobi','verbose',0);
sol = optimize(F,obj,opts); assert(sol.problem==0,sol.info);
xv = value(x); uv = value(u);

%% Build K, R at the active set, then C1 and rho
% Hessian H
H = zeros(nz,nz);
for t=0:T-1, H(xidx(t),xidx(t)) = 2*Q_vec(t+1); end
for t=0:T-1, H(uidx(t),uidx(t)) = 2*R_vec(t+1); end

% Equality Jacobian Meq z = b(x0,phi): dynamics, x0, xT
Meq = zeros(T+2, nz);
for t=0:T-1
    r=t+1; Meq(r,xidx(t+1))=1; Meq(r,xidx(t))=-1; Meq(r,uidx(t))=-1;
end
Meq(T+1,xidx(0))=1; Meq(T+2,xidx(T))=1;

% Active inequality Jacobian at baseline, and per-constraint meta
tol = 1e-9;
Aact = [];      % each row is ± e_k'
lab  = strings(0,1);
itime = [];     % time index i for the constraint (0..T for x, 0..T-1 for u)
for t=0:T-1
    if abs(uv(t+1)-umax) <= tol
        r=zeros(1,nz); r(1,uidx(t))=+1; Aact=[Aact;r]; lab=[lab; sprintf('u_%d<=umax',t)]; itime=[itime; t];
    end
    if abs(-uv(t+1)-umax) <= tol
        r=zeros(1,nz); r(1,uidx(t))=-1; Aact=[Aact;r]; lab=[lab; sprintf('-u_%d<=umax',t)]; itime=[itime; t];
    end
end
for t=1:T-1
    if abs(xv(t+1)-xmax) <= tol
        r=zeros(1,nz); r(1,xidx(t))=+1; Aact=[Aact;r]; lab=[lab; sprintf('x_%d<=xmax',t)]; itime=[itime; t];
    end
    if abs(-xv(t+1)-xmax) <= tol
        r=zeros(1,nz); r(1,xidx(t))=-1; Aact=[Aact;r]; lab=[lab; sprintf('-x_%d<=xmax',t)]; itime=[itime; t];
    end
end
M = [Meq; Aact]; m = size(M,1);

K = [H, M'; M, zeros(m,m)];

% Parameter map R = [S;N], here S=0, N has -1 in x0 and xT equality rows
R = zeros(nz+m, 2);
R(nz + (T+1), 1) = -1;   % ∂(x0 - p1)/∂p1
R(nz + (T+2), 2) = -1;   % ∂(xT - p2)/∂p2

% Spectral constants (bar and underline are max/min singular values)
sK = svd(K);  sigminK=min(sK); sigmaxK=max(sK);
sR = svd(R);  sigmaxR=max(sR);
C1  = (sigmaxK * sigmaxR) / (sigminK^2);
rho = (sigmaxK^2 - sigminK^2) / (sigmaxK^2 + sigminK^2);   % in (0,1)

%% Build diamond coefficients a_i, b_i for every inequality (use ALL, not only active)
% We want slacks of the *inequality family*, not only those currently tight.
% Order: [u_up(0..T-1); u_lo(0..T-1); x_up(1..T-1); x_lo(1..T-1)]
a = []; b = []; s = []; names = strings(0,1);
alpha = @(i,t) max(0, ceil( (abs(i - t) - 1)/2 ));  % with bandwidth b=1

% u bounds
for t=0:T-1
    s = [s;  umax - uv(t+1)]; names=[names; sprintf('u_%d<=',t)];
    a = [a;  C1 * rho^alpha(t,0) ];    b = [b; C1 * rho^alpha(t,T) ];
    s = [s;  umax + uv(t+1)]; names=[names; sprintf('-u_%d<=',t)];
    a = [a;  C1 * rho^alpha(t,0) ];    b = [b; C1 * rho^alpha(t,T) ];
end
% x bounds for t=1..T-1
for t=1:T-1
    s = [s;  xmax - xv(t+1)]; names=[names; sprintf('x_%d<=',t)];
    a = [a;  C1 * rho^alpha(t,0) ];    b = [b; C1 * rho^alpha(t,T) ];
    s = [s;  xmax + xv(t+1)]; names=[names; sprintf('-x_%d<=',t)];
    a = [a;  C1 * rho^alpha(t,0) ];    b = [b; C1 * rho^alpha(t,T) ];
end

% keep nonnegative slacks
s = max(s, 0);

%% Grid test, intersection of diamonds  a_i|dx0| + b_i|dφ| ≤ s_i
dx0 = linspace(-0.002, 0.002, 201);
dph = linspace(-0.002, 0.002, 201);
[DX0, DPH] = meshgrid(dx0, dph);

% evaluate all inequalities
feasible = true(size(DX0));
for i = 1:numel(s)
    feasible = feasible & ( a(i)*abs(DX0) + b(i)*abs(DPH) <= s(i) + 1e-12 );
end

%% Plot
figure; hold on; grid on;
contourf(DX0, DPH, feasible, [0.5 1.5], 'LineStyle','none'); colormap([0.9 0.95 1; 0.7 0.9 1]);
contour(DX0, DPH, feasible, [0.5 0.5], 'k','LineWidth',1.5);
xlabel('\delta x_0'); ylabel('\delta \phi');
title(sprintf('No-change region from decay bound,  C_1=%.2e, \\rho=%.4f', C1, rho));
axis equal tight;

%% Return info
info = struct();
info.C1 = C1; info.rho = rho;
info.a  = a;  info.b   = b; info.slack = s; info.names = names;
info.xopt = xv; info.uopt = uv;
end
