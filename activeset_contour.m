% Active-set robustness map over (x0, phi) using YALMIP + Gurobi
% 3-way classification: strong-active, weak-active, inactive
clear; clc;

%% Problem size and tunables
T = 10;

% Nominal pair for active-set comparison
x0_nom  = 0.0;
phi_nom = 0.0;

% Grids
x0_grid  = -0.3:0.01:-0.2;
phi_grid = -0.1:0.05:0.1;

% Bounds
xmax = 1.0;
umax = 1.0;

% Costs, scalar or length-T vectors
Q_in = 1.0;    % state weights t=0..T-1
R_in = 1.0;    % input weights t=0..T-1

% Time-varying reference via block repeat
xi_block = [1, -1, 2, 0.7];
xi_vec = repmat(xi_block(:), ceil(T/numel(xi_block)), 1);
xi_vec = xi_vec(1:T);

% Normalize Q, R
Q_vec = isscalar(Q_in)*repmat(Q_in,T,1) + (~isscalar(Q_in))*Q_in(:);
R_vec = isscalar(R_in)*repmat(R_in,T,1) + (~isscalar(R_in))*R_in(:);
assert(numel(Q_vec)==T && numel(R_vec)==T);

%% Decision vars
x = sdpvar(T+1,1);    % x0..xT
u = sdpvar(T,1);      % u0..u_{T-1}

%% Objective
objective = sum( Q_vec .* (x(1:T) - xi_vec).^2 ) + sum( R_vec .* (u.^2) );

%% Static constraints, independent of x0, phi
F_static = [];

% Dynamics
dyn = cell(T,1);
for t = 1:T
    c = (x(t+1) == x(t) + u(t));
    F_static = [F_static, c];
    dyn{t} = c; %#ok<NASGU>
end

% Input bounds
ub_u = cell(T,1);  lb_u = cell(T,1);
for t = 1:T
    c1 = (  u(t) <=  umax );  ub_u{t} = c1;
    c2 = ( -u(t) <=  umax );  lb_u{t} = c2;
    F_static = [F_static, c1, c2];
end

% State bounds for interior states, t = 1..T-1  (x(2)..x(T))
ub_x = cell(T-1,1);  lb_x = cell(T-1,1);
for t = 2:T
    c1 = (  x(t) <=  xmax );  ub_x{t-1} = c1;
    c2 = ( -x(t) <=  xmax );  lb_x{t-1} = c2;
    F_static = [F_static, c1, c2];
end
% Inequalities total = 4T-2

%% Gurobi options, tightened tolerances, no tuning
opts = sdpsettings('solver','gurobi','verbose',0, ...
    'gurobi.FeasibilityTol', 1e-9, ...
    'gurobi.OptimalityTol',  1e-9, ...
    'gurobi.BarConvTol',     1e-12, ...
    'gurobi.NumericFocus',   2, ...
    'gurobi.TuneTimeLimit',  0, ...
    'gurobi.TuneResults',    0, ...
    'gurobi.TuneOutput',     0);

%% Classification thresholds
eps_p = 1e-6;   % primal threshold for |g| near boundary
eps_d = 1e-6;   % dual threshold for “non-tiny” multipliers

%% Helper, solve at numeric (x0,phi), return 3-way inequality pattern
function [status_vec, xval, uval, lambda_ineq, g_vals] = ...
    solve_active_numeric(x0val, phival, F_static, objective, opts, x, u, umax, xmax, ub_u, lb_u, ub_x, lb_x, T, eps_p, eps_d)

    % Rebuild the two endpoint equalities numerically
    F_eq = [ x(1) == x0val, x(T+1) == phival ];
    F = [F_static, F_eq];

    sol = optimize(F, objective, opts);
    if sol.problem ~= 0
        error('Solve failed, problem=%d, info=%s', sol.problem, sol.info);
    end

    xval = value(x); uval = value(u);

    % Duals in construction order, inequalities only
    lambda_ineq = [];
    for tt = 1:numel(ub_u)
        lambda_ineq = [lambda_ineq; dual(ub_u{tt}); dual(lb_u{tt})]; %#ok<AGROW>
    end
    for tt = 1:numel(ub_x)
        lambda_ineq = [lambda_ineq; dual(ub_x{tt}); dual(lb_x{tt})]; %#ok<AGROW>
    end
    lambda_ineq = lambda_ineq(:);

    % Residuals g(z) = A z - b, same order
    g_vals = [];
    for tt = 1:numel(ub_u)
        g_vals = [g_vals;  uval(tt) - umax; -uval(tt) - umax]; %#ok<AGROW>
    end
    for tt = 1:numel(ub_x)
        g_vals = [g_vals;  xval(tt+1) - xmax; -xval(tt+1) - xmax]; %#ok<AGROW>
    end
    g_vals = g_vals(:);

    % 3-way classification
    is_near_bd = abs(g_vals) <= eps_p;
    has_dual   = lambda_ineq >= eps_d;

    % status: 0 = inactive, 1 = weak-active, 2 = strong-active
    status_vec = zeros(size(g_vals));               % inactive
    status_vec(is_near_bd & ~has_dual) = 1;         % weak
    status_vec(is_near_bd &  has_dual) = 2;         % strong

    % if clearly interior, keep as inactive, even if tiny positive lambda
    % i.e., g < -eps_p implies inactive regardless of lambda
    status_vec(g_vals < -eps_p) = 0;
end

%% Baseline 3-way pattern at nominal pair
[status_nom, ~, ~, ~, ~] = solve_active_numeric( ...
    x0_nom, phi_nom, F_static, objective, opts, x, u, umax, xmax, ub_u, lb_u, ub_x, lb_x, T, eps_p, eps_d);

%% Sweep grid, compare 3-way patterns
nx0 = numel(x0_grid); nphi = numel(phi_grid);
sameAS = false(nx0, nphi);

for i = 1:nx0
    for j = 1:nphi
        [status_ij, ~, ~, ~, ~] = solve_active_numeric( ...
            x0_grid(i), phi_grid(j), F_static, objective, opts, x, u, umax, xmax, ub_u, lb_u, ub_x, lb_x, T, eps_p, eps_d);
        sameAS(i,j) = isequal(status_ij, status_nom);
    end
end

%% Plot, circles where 3-way pattern matches, contour boundary
[X0, PHI] = meshgrid(x0_grid, phi_grid);   % size nphi x nx0
sameAS_T = sameAS';                        % transpose to match meshgrid

figure; hold on; grid on;
[idx_phi, idx_x0] = find(sameAS_T);
plot(x0_grid(idx_x0), phi_grid(idx_phi), 'o', 'MarkerSize', 4, 'LineWidth', 1.5);
contour(X0, PHI, double(sameAS_T), [0.5 0.5], 'LineWidth', 1.5);
plot(x0_nom, phi_nom, 'vr', 'MarkerSize', 6,  'LineWidth', 2)

xlabel('x_0'); ylabel('\phi');
title('Region where 3-way inequality pattern matches the nominal');
legend('Same 3-way pattern', 'Contour boundary', 'Nominal parameter', 'Location', 'best');
axis tight;
