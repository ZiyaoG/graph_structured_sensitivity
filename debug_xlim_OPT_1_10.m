function [x_opt, J] = run_full_opt_hq(T_sim, x0, ulim, xlim, xi_full, u_flag, use_eps)
if nargin < 7, use_eps = false; end

x = sdpvar(T_sim,1);
obj = 0; cons = [];
for t = 1:T_sim
    obj = obj + (x(t)-xi_full(t))^2;
    if t==1
        obj = obj + u_flag*(x(1)-x0)^2;
        cons = [cons;  x(1)-x0 <= ulim;  x0-x(1) <= ulim];
    else
        obj = obj + u_flag*(x(t)-x(t-1))^2;
        cons = [cons;  x(t)-x(t-1) <= ulim;  x(t-1)-x(t) <= ulim];
    end
    cons = [cons; x(t) <= xlim; -x(t) <= xlim];
end

% Optional tiny tie-breaker to remove degeneracy
if use_eps
    epsilon = 1e-10;                % try 1e-10 to 1e-8
    obj = obj + epsilon*(x.'*x);    % strictly convex, doesn’t change behavior
end

opts = sdpsettings('solver','gurobi','verbose',0,'cachesolvers',1);

% Deterministic, high-precision, and disable the tuner that caused your error
opts.gurobi.Threads        = 1;
opts.gurobi.Method         = 2;      % barrier
opts.gurobi.Crossover      = 0;      % no crossover
opts.gurobi.NumericFocus   = 3;
opts.gurobi.ScaleFlag      = 2;
opts.gurobi.Presolve       = 1;
opts.gurobi.FeasibilityTol = 1e-9;
opts.gurobi.OptimalityTol  = 1e-9;
opts.gurobi.BarConvTol     = 1e-12;
opts.gurobi.TuneTimeLimit  = 0;      % <-- fixes "value -1" error

sol = optimize(cons, obj, opts);
assert(sol.problem==0, sol.info);

x_opt = value(x);
J     = value(obj);
end



% build reference
T_sim = 500; x0 = 0; ulim = 0.8; xi = 0.8; u_flag = 0;
xi_full = repmat(xi, T_sim, 1); xi_full(2:2:end) = -xi;

% without tie-breaker
[x1,  J1 ] = run_full_opt_hq(T_sim, x0, ulim, 1,  xi_full, u_flag, false);
[x10, J10] = run_full_opt_hq(T_sim, x0, ulim, 10, xi_full, u_flag, false);
fprintf('no-ε: max|Δx| = %.3e, ΔJ = %.3e\n', max(abs(x1-x10)), J1-J10);

% % with tie-breaker  % no difference compared to without tie-breaker
% [x1e,  J1e ] = run_full_opt_hq(T_sim, x0, ulim, 1,  xi_full, u_flag, true);
% [x10e, J10e] = run_full_opt_hq(T_sim, x0, ulim, 10, xi_full, u_flag, true);
% fprintf(' with ε: max|Δx| = %.3e, ΔJ = %.3e\n', max(abs(x1e-x10e)), J1e-J10e);
