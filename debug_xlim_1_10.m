clear; close all;

%% -------------------- SHARED PARAMETERS --------------------
T_sim   = 500;          % total sim steps
Np      = 30;           % MPC horizon
x0      = 0;            % initial state
R       = 0.1;          % terminal tube radius
ulim    = 4/5;          % move limit
xi      = 4/5;          % alternating reference (+/-)
u_flag  = 0;            % 1 = move penalty, 0 = constrain only
xlim_list = [1, 10];    % we will run matched OPT and MPC for each

xi_full = repmat(xi, T_sim, 1);  xi_full(2:2:end) = -xi;
% options = sdpsettings('solver','gurobi','verbose',0);
options = sdpsettings('solver','gurobi','verbose',0,'cachesolvers',1);
options.gurobi.Threads        = 1;    % deterministic
options.gurobi.Method         = 2;    % barrier
options.gurobi.Crossover      = 0;    % no crossover perturbations
options.gurobi.NumericFocus   = 3;    % be conservative numerically
options.gurobi.ScaleFlag      = 2;    % aggressive scaling
options.gurobi.Presolve       = 1;    % gentler presolve (helps reproducibility)
options.gurobi.FeasibilityTol = 1e-9;
options.gurobi.OptimalityTol  = 1e-9;
options.gurobi.BarConvTol     = 1e-12;
options.gurobi.TuneTimeLimit  = 0;      % <-- fixes "value -1" error

%% -------------------- RUN OPT FOR EACH XLIM --------------------
OPT = struct('x',[],'x_ext',[],'cost',[],'xlim',[],'maxabs',[]);
for j = 1:2
    xlim = xlim_list(j);
    [x_opt, cost_opt] = run_full_opt(T_sim, x0, ulim, xlim, xi_full, u_flag, options);
    OPT(j).x      = x_opt;
    OPT(j).x_ext  = [x_opt; x_opt(end)];
    OPT(j).cost   = cost_opt;
    OPT(j).xlim   = xlim;
    OPT(j).maxabs = max(abs(x_opt));
end

%% -------------------- BUILD MPC BANKS, ONE PER XLIM --------------------
F_bank = cell(2,1);
for j = 1:2
    F_bank{j} = build_mpc_bank(xlim_list(j), Np, ulim, R, u_flag, options);
end

%% -------------------- SIMULATE TWO MPC TRAJECTORIES --------------------
MPC = struct('x',[],'cost',[],'xlim',[]);
for j = 1:2
    x_sim = zeros(T_sim+1,1); x_sim(1) = x0;
    for k = 1:T_sim
        h = min(Np, T_sim-k+1);
        xi_pred  = xi_full(k : k+h-1);
        p_term_k = OPT(j).x_ext(k+h);           % matched OPT terminal target
        x_pred   = F_bank{j}{h}{[ x_sim(k); xi_pred; p_term_k ]};
        x_sim(k+1) = x_pred(1);
    end
    tracking_err = x_sim(2:end) - xi_full;
    control_eff  = diff(x_sim);
    MPC(j).x     = x_sim;
    MPC(j).cost  = sum(tracking_err.^2) + u_flag*sum(control_eff.^2);
    MPC(j).xlim  = xlim_list(j);
end

%% -------------------- REPORT --------------------
fprintf('FULL-HORIZON OPT costs and bound usage:\n');
for j = 1:2
    fprintf('  OPT with xlim=%2d:  cost = %.6f,  max|x| = %.6f %s xlim\n', ...
        OPT(j).xlim, OPT(j).cost, OPT(j).maxabs, cond_word(OPT(j).maxabs, OPT(j).xlim));
end
fprintf('\nMPC costs using matched OPT terminal targets:\n');
for j = 1:2
    fprintf('  MPC with xlim=%2d:  cost = %.6f\n', MPC(j).xlim, MPC(j).cost);
end
fprintf('  Cost difference, xlim=1 minus xlim=10: %.6f\n', MPC(1).cost - MPC(2).cost);

%% -------------------- PLOTS --------------------
t = 0:T_sim;

figure(1); clf
plot(t, MPC(1).x, 'LineWidth', 1.5); hold on
plot(t, MPC(2).x, 'LineWidth', 1.5);
plot(1:T_sim, xi_full, 'k--', 'LineWidth', 1.0);
yline( 1, ':'); yline(-1, ':'); yline(10, ':'); yline(-10, ':');
grid on; xlabel('time step'); ylabel('state x');
legend('MPC, x_{max}=1','MPC, x_{max}=10','\xi_t','\pm1','\pm10','Location','best');
title('MPC trajectories, each using its matched OPT terminal reference');

figure(2); clf
plot(t, MPC(1).x - MPC(2).x, 'LineWidth', 1.5);
grid on; xlabel('time step'); ylabel('\Delta x = x_{xlim=1} - x_{xlim=10}');
title('Trajectory difference between the two MPC runs');

%% ==================== FUNCTIONS ====================
function [x_opt_value, cost_opt] = run_full_opt(T_sim, x0, ulim, xlim, xi_full, u_flag, options)
    x_opt = sdpvar(T_sim,1);
    cons  = [];
    obj   = 0;
    for t = 1:T_sim
        obj = obj + (x_opt(t)-xi_full(t))^2;
        if t==1
            obj = obj + u_flag*(x_opt(1)-x0)^2;
            cons = [cons, x_opt(1)-x0 <= ulim, x0 - x_opt(1) <= ulim];
        else
            obj = obj + u_flag*(x_opt(t)-x_opt(t-1))^2;
            cons = [cons, x_opt(t)-x_opt(t-1) <= ulim, x_opt(t-1)-x_opt(t) <= ulim];
        end
        cons = [cons, -xlim <= x_opt(t) <= xlim];
    end
    sol = optimize(cons, obj, options);
    assert(sol.problem==0, 'FULL-OPT failed: %s', yalmiperror(sol.problem));
    x_opt_value = value(x_opt);
    cost_opt    = value(obj);
end

function F = build_mpc_bank(xlim, Np, ulim, R, u_flag, options)
    F = cell(Np,1);
    for h = 1:Np
        x_seq = sdpvar(h+1,1);   % predicted states
        p0    = sdpvar(1,1);     % current state
        p_xi  = sdpvar(h,1);     % reference over horizon
        p_term= sdpvar(1,1);     % terminal target from matched OPT

        cons  = [];
        obj   = 0;

        for t = 1:h
            obj  = obj + (x_seq(t)-p_xi(t))^2;
            cons = [cons, -xlim <= x_seq(t) <= xlim];
        end
        for t = 1:h+1
            if t==1
                cons = [cons, -ulim <= x_seq(1)-p0 <= ulim];
                obj  = obj + u_flag*(x_seq(1)-p0)^2;
            else
                cons = [cons, -ulim <= x_seq(t)-x_seq(t-1) <= ulim];
                obj  = obj + u_flag*(x_seq(t)-x_seq(t-1))^2;
            end
        end
        cons = [cons, p_term - R <= x_seq(h+1) <= p_term + R];

        F{h} = optimizer(cons, obj, options, [p0; p_xi; p_term], x_seq);
    end
end

function s = cond_word(maxabsx, xlim)
    if maxabsx >= xlim - 1e-8
        s = '(touches or binds)';
    else
        s = '(strictly inside)';
    end
end
