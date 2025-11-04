% compare_primals_duals_with_classes.m
% Primals, duals, residuals, and 3-way active-set classification
% for (x0,phi) = (0,-0.3) and (0,-0.4)

clear; clc;

%% Problem size and tunables
T = 10;

% Two test pairs
pairs = [ -0.25, 0;
          -0.26, 0 ];

% Bounds
xmax = 1.0;
umax = 1.0;

% Costs
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
    dyn{t} = c;
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

%% Gurobi options, tightened tolerances, tuning disabled
opts = sdpsettings('solver','gurobi','verbose',0, ...
    'gurobi.FeasibilityTol', 1e-9, ...
    'gurobi.OptimalityTol',  1e-9, ...
    'gurobi.BarConvTol',     1e-12, ...
    'gurobi.NumericFocus',   2, ...
    'gurobi.TuneTimeLimit',  0);     % important: avoid -1 default

%% Classification thresholds
eps_p = 1e-6;   % for |g| near the boundary
eps_d = 1e-6;   % for “non-tiny” multipliers

%% Helper: safe optimize
function safe_optimize(F, obj, opts)
    sol = optimize(F, obj, opts);
    if sol.problem ~= 0
        error('Gurobi/YALMIP reported problem=%d: %s', sol.problem, sol.info);
    end
end

%% Helper: solve and collect primals, duals, residuals
function S = solve_point(x0val, phival, F_static, objective, opts, x, u, umax, xmax, ub_u, lb_u, ub_x, lb_x, dyn, T)
    eq_x0  = (x(1)   == x0val);
    eq_phi = (x(T+1) == phival);
    F = [F_static, eq_x0, eq_phi];

    safe_optimize(F, objective, opts);

    S.x = value(x);
    S.u = value(u);

    % Duals, endpoint equalities
    S.lambda_eq.x0  = dual(eq_x0);
    S.lambda_eq.phi = dual(eq_phi);

    % Duals, dynamics equalities
    S.lambda_dyn = zeros(T,1);
    for tt = 1:T
        S.lambda_dyn(tt) = dual(dyn{tt});
    end

    % Inequality duals in build order
    lam_ineq = [];
    for tt = 1:numel(ub_u)
        lam_ineq = [lam_ineq; dual(ub_u{tt}); dual(lb_u{tt})]; %#ok<AGROW>
    end
    for tt = 1:numel(ub_x)
        lam_ineq = [lam_ineq; dual(ub_x{tt}); dual(lb_x{tt})]; %#ok<AGROW>
    end
    S.lambda_ineq = lam_ineq(:);

    % Residuals g(z) in the same order
    g_vals = [];
    for tt = 1:numel(ub_u)
        g_vals = [g_vals;  S.u(tt) - umax; -S.u(tt) - umax]; %#ok<AGROW>
    end
    for tt = 1:numel(ub_x)
        g_vals = [g_vals;  S.x(tt+1) - xmax; -S.x(tt+1) - xmax]; %#ok<AGROW>
    end
    S.g_ineq = g_vals(:);
end

%% Helper: 3-way classification
function C = classify_constraints(S, eps_p, eps_d, T)
    C.is_near_bd = abs(S.g_ineq) <= eps_p;
    C.has_dual   = S.lambda_ineq >= eps_d;

    C.active_strong = C.is_near_bd & C.has_dual;
    C.active_weak   = C.is_near_bd & ~C.has_dual;
    C.inactive      = (S.g_ineq < -eps_p) | (~C.is_near_bd & ~C.has_dual);

    status = repmat("inactive", size(S.g_ineq));
    status(C.active_weak)   = "weak-active";
    status(C.active_strong) = "strong-active";
    C.status = status;

    C.idx_u_ub  = (1:2:2*T).';
    C.idx_u_lb  = (2:2:2*T).';
    C.idx_x_ub  = (2*T+1:2:2*T+2*(T-1)).';
    C.idx_x_lb  = (2*T+2:2:2*T+2*(T-1)).';

    C.count_strong = sum(C.active_strong);
    C.count_weak   = sum(C.active_weak);
    C.count_inact  = sum(C.inactive);
end

%% Pretty print
function print_summary(tag, S, C, T)
    fprintf('\n================= %s =================\n', tag);
    fprintf('Endpoint duals: lambda_x0 = %+ .4e, lambda_phi = %+ .4e\n', S.lambda_eq.x0, S.lambda_eq.phi);

    fprintf('\nStates x(0..T):\n');
    fprintf('t      x(t)\n');
    for t = 0:T
        fprintf('%2d   %+ .6f\n', t, S.x(t+1));
    end

    fprintf('\nInputs u(0..T-1):\n');
    fprintf('t      u(t)\n');
    for t = 0:T-1
        fprintf('%2d   %+ .6f\n', t, S.u(t+1));
    end

    fprintf('\nInequality duals, slacks, and status:\n');
    fprintf('Type      t     lambda         g(z)         status\n');

    % u bounds
    for t = 1:T
        k = 2*(t-1)+1;
        fprintf('u<=umax  %2d  %+ .6e  %+ .6e     %s\n', t-1, S.lambda_ineq(k),   S.g_ineq(k),   C.status(k));
        fprintf('-u<=umax %2d  %+ .6e  %+ .6e     %s\n', t-1, S.lambda_ineq(k+1), S.g_ineq(k+1), C.status(k+1));
    end

    % x bounds, t=1..T-1 relate to x indices 2..T
    base = 2*T;
    for t = 1:T-1
        k = base + 2*(t-1) + 1;
        fprintf('x<=xmax  %2d  %+ .6e  %+ .6e     %s\n', t, S.lambda_ineq(k),   S.g_ineq(k),   C.status(k));
        fprintf('-x<=xmax %2d  %+ .6e  %+ .6e     %s\n', t, S.lambda_ineq(k+1), S.g_ineq(k+1), C.status(k+1));
    end

    comp = S.lambda_ineq .* S.g_ineq;
    fprintf('\nComplementarity:  max |lambda.*g| = %.3e,  min(lambda.*g) = %.3e\n', ...
        max(abs(comp)), min(comp));
    fprintf('Counts: strong-active = %d, weak-active = %d, inactive = %d  (total = %d)\n', ...
        C.count_strong, C.count_weak, C.count_inact, numel(C.status));

    fprintf('\nDynamics duals (x(t+1)=x(t)+u(t)):\n');
    fprintf('t    lambda_dyn\n');
    for t = 1:T
        fprintf('%2d   %+ .6e\n', t-1, S.lambda_dyn(t));
    end
end

%% Plot
function plot_trajectories(S1, S2, tag1, tag2, T)
    figure; 
    subplot(2,1,1); hold on; grid on;
    stairs(0:T,   S1.x, 'o-'); 
    stairs(0:T,   S2.x, 's-'); 
    xlabel('t'); ylabel('x_t'); title('States');
    legend(tag1, tag2, 'Location', 'best');

    subplot(2,1,2); hold on; grid on;
    stairs(0:T-1, S1.u, 'o-');
    stairs(0:T-1, S2.u, 's-');
    xlabel('t'); ylabel('u_t'); title('Inputs');
    legend(tag1, tag2, 'Location', 'best');
end

%% Solve, classify, print
S1 = solve_point(pairs(1,1), pairs(1,2), F_static, objective, opts, x, u, umax, xmax, ub_u, lb_u, ub_x, lb_x, dyn, T);
C1 = classify_constraints(S1, eps_p, eps_d, T);
print_summary(sprintf('(x0,phi) = (%.1f, %.1f)', pairs(1,1), pairs(1,2)), S1, C1, T);

S2 = solve_point(pairs(2,1), pairs(2,2), F_static, objective, opts, x, u, umax, xmax, ub_u, lb_u, ub_x, lb_x, dyn, T);
C2 = classify_constraints(S2, eps_p, eps_d, T);
print_summary(sprintf('(x0,phi) = (%.1f, %.1f)', pairs(2,1), pairs(2,2)), S2, C2, T);

plot_trajectories(S1, S2, '(0,-0.3)', '(0,-0.4)', T);
