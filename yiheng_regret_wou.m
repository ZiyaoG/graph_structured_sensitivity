% Temporal MPC vs Full-Horizon OPT Regret Analysis
% Computes total cost of MPC with horizons N=1:10 and compares to full T-step optimum

%% PARAMETERS
T_sim   = 100;         % Total simulation steps
Np_max  = 10;          % Maximum MPC horizon to test (horizons 1..10)
x0      = 0;           % Initial state

% Build full reference signal xi_full for simulation
xi_full = repmat(4/5, T_sim, 1);
xi_full(2:2:end) = -4/5;

% Preallocate storage
cost_mpc = zeros(Np_max,1);

% Solver settings for YALMIP
options = sdpsettings('solver','gurobi','verbose',0);

%% MPC SIMULATIONS FOR Np = 1..Np_max
for Np = 1:Np_max
    % Construct parametric optimizer for current horizon
    x_seq = sdpvar(Np,1);
    p0    = sdpvar(1,1);
    p_xi  = sdpvar(Np,1);
    cons  = [];
    obj   = 0;
    for t = 1:Np
        obj = obj + (x_seq(t) - p_xi(t))^2;
        cons = [cons, -1 <= x_seq(t) <= 1];
        if t==1
            cons = [cons, x_seq(1)-p0 <= 4/5, p0-x_seq(1) <= 4/5];
        else
            cons = [cons, x_seq(t)-x_seq(t-1) <= 4/5, x_seq(t-1)-x_seq(t) <= 4/5];
        end
    end
    F = optimizer(cons, obj, options, [p0; p_xi], x_seq);

    % Run receding-horizon MPC
    x_sim = zeros(T_sim+1,1);
    x_sim(1) = x0;
    for k = 1:T_sim
        xi_pred = xi_full(k:min(k+Np-1,T_sim));
        % If at end, pad reference (won't matter beyond T_sim)
        if numel(xi_pred)<Np
            xi_pred = [xi_pred; zeros(Np-numel(xi_pred),1)];
        end
        params = [x_sim(k); xi_pred];
        x_pred = F{params};
        x_sim(k+1) = x_pred(1);
    end

    % Compute total cost for MPC
    errors = x_sim(2:end) - xi_full;
    cost_mpc(Np) = sum(errors.^2);
end

%% FULL-HORIZON OPTIMIZATION (GROUND TRUTH)
x_opt = sdpvar(T_sim,1);
cons_opt = [];
obj_opt   = 0;
for t = 1:T_sim
    obj_opt = obj_opt + (x_opt(t) - xi_full(t))^2;
    cons_opt = [cons_opt, -1 <= x_opt(t) <= 1];
    if t == 1
        cons_opt = [cons_opt, x_opt(1)-x0 <= 4/5, x0-x_opt(1) <= 4/5];
    else
        cons_opt = [cons_opt, x_opt(t)-x_opt(t-1) <= 4/5, x_opt(t-1)-x_opt(t) <= 4/5];
    end
end
opt_sol = optimize(cons_opt, obj_opt, options);
if opt_sol.problem ~= 0
    error('Full-horizon optimization failed: %s', opt_sol.info);
end
cost_opt = value(obj_opt);

%% PLOT REGRET vs Horizon
horizons = (1:Np_max)';
regret   = cost_mpc - cost_opt;

figure;
loglog(horizons, regret, '-o', 'LineWidth',1.5);
xlabel('MPC Horizon N');
ylabel('Regret (Cost_{MPC}-Cost_{OPT})');
title('Regret vs Horizon for Temporal MPC');
grid on;

% Also display numeric summary
fprintf('Full-horizon OPT cost: %.4f\n', cost_opt);
for Np = 1:Np_max
    fprintf('Horizon %2d: MPC cost = %.4f, Regret = %.4f\n', Np, cost_mpc(Np), regret(Np));
end
