function onestep_diff_via_OPT_terminal_gurobi
clear; close all;

%% Shared params
T_sim = 500;      % same as your run
Np    = 30;       % MPC horizon length
x0    = 0.0;
ulim  = 0.8;
R     = 0.10;
xi    = 0.80;     % keep exactly 0.8
u_flag= 0;

% alternating reference
xi_full = repmat(xi, T_sim, 1);
xi_full(2:2:end) = -xi;

% GUROBI only
% opts = sdpsettings('solver','gurobi','verbose',0,'cachesolvers',1);
opts = sdpsettings('solver','gurobi','verbose',0,'cachesolvers',1);
opts.gurobi.Threads        = 1;    % deterministic
opts.gurobi.Method         = 2;    % barrier
opts.gurobi.Crossover      = 0;    % no crossover perturbations
opts.gurobi.NumericFocus   = 3;    % be conservative numerically
opts.gurobi.ScaleFlag      = 2;    % aggressive scaling
opts.gurobi.Presolve       = 1;    % gentler presolve (helps reproducibility)
opts.gurobi.FeasibilityTol = 1e-9;
opts.gurobi.OptimalityTol  = 1e-9;
opts.gurobi.BarConvTol     = 1e-12;
opts.gurobi.TuneTimeLimit  = 0;      % <-- fixes "value -1" error

%% 1) Solve two OPTs with matched xlim
xlims = [1, 10];
OPT   = struct('x',[],'xext',[],'cost',[],'xlim',[]);
for j = 1:2
    xlim = xlims(j);
    [x_opt, Jopt] = run_full_opt(T_sim, x0, ulim, xlim, xi_full, u_flag, opts);
    OPT(j).x    = x_opt;
    OPT(j).xext = [x_opt; x_opt(end)];
    OPT(j).cost = Jopt;
    OPT(j).xlim = xlim;
end

% Show whether bounds touched in OPT
fprintf('FULL OPT summary:\n');
for j = 1:2
    fprintf('  xlim=%2d: cost=%.6f, max|x|=%.6f\n', OPT(j).xlim, OPT(j).cost, max(abs(OPT(j).x)));
end

%% 2) Extract the matched terminal targets for the first MPC call
% first MPC call uses horizon h = min(Np, T_sim-0) = Np, so p_term = x_opt(Np+1)
pterm_1  = OPT(1).xext(Np+1);   % for xlim=1
pterm_10 = OPT(2).xext(Np+1);   % for xlim=10

%% 3) Solve one-step MPC twice, same xi and x0, different p_term and xlim
x1  = onestep_mpc(x0, xi, pterm_1,  R, ulim, xlims(1), opts);
x10 = onestep_mpc(x0, xi, pterm_10, R, ulim, xlims(2), opts);

% Analytic projections to verify numerics
xA1  = analytic_projection(x0, xi, pterm_1,  R, ulim, xlims(1));
xA10 = analytic_projection(x0, xi, pterm_10, R, ulim, xlims(2));

%% 4) Report
fprintf('\nONE-STEP MPC with matched OPT terminals (xi=%.6f):\n', xi);
fprintf('  xlim= 1:  x* = %.9f,  projection = %.9f,  |diff|=%.2e\n', x1,  xA1,  abs(x1 - xA1));
fprintf('  xlim=10:  x* = %.9f,  projection = %.9f,  |diff|=%.2e\n', x10, xA10, abs(x10 - xA10));
fprintf('  Δx* = x*(xlim=1) - x*(xlim=10) = %.9e\n', x1 - x10);
fprintf('  Δp_term = %.9e  (xopt_{Np+1}^{xlim=1} - xopt_{Np+1}^{xlim=10})\n', pterm_1 - pterm_10);

end

%% ---- helpers ----
function [x_opt_value, cost_opt] = run_full_opt(T_sim, x0, ulim, xlim, xi_full, u_flag, opts)
    x_opt = sdpvar(T_sim,1);
    cons  = [];
    obj   = 0;
    for t = 1:T_sim
        obj = obj + (x_opt(t)-xi_full(t))^2;
        if t==1
            obj = obj + u_flag*(x_opt(1)-x0)^2;
            cons = [cons; x_opt(1)-x0 <= ulim; x0 - x_opt(1) <= ulim];
        else
            obj = obj + u_flag*(x_opt(t)-x_opt(t-1))^2;
            cons = [cons; x_opt(t)-x_opt(t-1) <= ulim; x_opt(t-1)-x_opt(t) <= ulim];
        end
        cons = [cons; x_opt(t) <= xlim; -x_opt(t) <= xlim];
    end
    sol = optimize(cons, obj, opts);
    if sol.problem ~= 0
        error('OPT failed: %s', sol.info);
    end
    x_opt_value = value(x_opt);
    cost_opt    = value(obj);
end

function xstar = onestep_mpc(x0, xi, pterm, R, ulim, xlim, opts)
    x = sdpvar(1,1);
    obj = (x - xi)^2;
    cons = [ x <=  xlim;  -x <=  xlim; ...
             x - x0 <=  ulim;  x0 - x <=  ulim; ...
             x <=  pterm + R;  -x <= -(pterm - R) ];
    a = max([-xlim, x0-ulim, pterm-R]);
    b = min([ xlim, x0+ulim, pterm+R]);
    if a > b + 1e-12, error('One-step infeasible: [%g,%g]', a, b); end
    sol = optimize(cons, obj, opts);
    if sol.problem ~= 0
        error('One-step failed: %s', sol.info);
    end
    xstar = value(x);
end

function xproj = analytic_projection(x0, xi, pterm, R, ulim, xlim)
    a = max([-xlim, x0-ulim, pterm-R]);
    b = min([ xlim, x0+ulim, pterm+R]);
    xproj = min(max(xi, a), b);
end
