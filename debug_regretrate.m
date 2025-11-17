clear; close all;

%% PARAMETERS
T_sim   = 500;         % Total sim steps
Np_max  = 19;          % Max MPC horizon to test
x0      = 0;           % Initial state
R       = 5;
xlim    = 1e4;         % never reached
ulim    = 4/5;
xi      = 4/5;

xi_full = repmat(xi, T_sim, 1);
xi_full(2:2:end) = -xi;
% Unreachable ramp reference: slope > u_max
% r = 1.1 * ulim;                  % slope > u_max
% t_vec = (1:T_sim)';              
% xi_full = r * t_vec;             % pure ramp

u_flag = 0;

options = sdpsettings('solver','gurobi','verbose',0);

%% FULL-HORIZON OPT
x_opt    = sdpvar(T_sim,1);
cons_opt = [];
obj_opt  = 0;
for t = 1:T_sim
    obj_opt = obj_opt + (x_opt(t)-xi_full(t))^2;
    
    if t==1
        obj_opt = obj_opt + u_flag*(x_opt(1)-x0)^2;
        cons_opt = [cons_opt, x_opt(1)-x0 <= ulim, x0-x_opt(1) <= ulim];
    else
        obj_opt = obj_opt + u_flag*(x_opt(t)-x_opt(t-1))^2;
        cons_opt = [cons_opt, x_opt(t)-x_opt(t-1) <= ulim, ...
                             x_opt(t-1)-x_opt(t) <= ulim];
    end
    cons_opt = [cons_opt, -xlim <= x_opt(t) <= xlim];
end

opt_sol = optimize(cons_opt, obj_opt, options);
if opt_sol.problem
    error('FULL-OPT failed with Gurobi: %s', yalmiperror(opt_sol.problem));
end
cost_opt    = value(obj_opt);
x_opt_value = value(x_opt);

%% PRE-BUILD OPTIMIZERS FOR ALL HORIZONS h=1..Np_max
F_all = cell(Np_max,1);
for h = 1:Np_max
    x_seq  = sdpvar(h+1,1);
    p0     = sdpvar(1,1);
    p_xi   = sdpvar(h,1);
    p_term = sdpvar(1,1);    % NEW: terminal tube center (time-varying)
    cons   = [];
    obj    = 0;

    for t = 1:h
        obj = obj + (x_seq(t)-p_xi(t))^2;           % tracking
        if t==1
            obj  = obj + u_flag*(x_seq(1)-p0)^2;    % effort at first step
            cons = [cons, x_seq(1)-p0 <= ulim, p0 - x_seq(1) <= ulim];
        else
            obj  = obj + u_flag*(x_seq(t)-x_seq(t-1))^2;  % effort
            cons = [cons, x_seq(t)-x_seq(t-1) <= ulim, x_seq(t-1)-x_seq(t) <= ulim];
        end
        cons = [cons, -xlim <= x_seq(t) <= xlim];   % state bounds for 1..h
    end

    % Couple final state (t = h+1) to x(h) via move limit
    obj  = obj + u_flag*(x_seq(h+1)-x_seq(h))^2;
    cons = [cons, x_seq(h+1)-x_seq(h) <= ulim, x_seq(h)-x_seq(h+1) <= ulim];

    % Terminal tube around *time-varying* p_term (center provided at runtime)
    cons = [cons, p_term - R <= x_seq(h+1) <= p_term + R];

    % Optimizer now parametrized by [p0; p_xi; p_term]
    F_all{h} = optimizer(cons, obj, options, [p0; p_xi; p_term], x_seq);
end

%% MPC SIMULATIONS FOR Np = 1..Np_max
cost_mpc     = zeros(Np_max,1);
TrajDev      = cell(Np_max,1);   % x_sim - x_opt_value (per horizon)
TrajDevMean  = NaN(Np_max,1);    % mean |x_sim - x_opt|
CenterMean   = NaN(Np_max,1);    % mean |x_mpc + x_opt - 2*xi|

for Np = 1:Np_max
    x_sim = zeros(T_sim+1,1);
    x_sim(1) = x0;
    
    for k = 1:T_sim
        % shrink horizon if near end
        Nloc = min(Np, T_sim-k+1);
        
        % grab the matching optimizer
        F = F_all{Nloc};
        
        % the matching reference slice
        xi_pred = xi_full(k : k+Nloc-1);

        % terminal tube center = clairvoyant x_opt at time k+Nloc
        term_idx = min(k+Nloc, T_sim);          % clamp to T_sim
        p_term_k = x_opt_value(term_idx);

        % solve
        x_pred = F{[ x_sim(k); xi_pred; p_term_k ]};
        
        % apply first move
        x_sim(k+1) = x_pred(1);
    end
    
    % compute total cost
    tracking_err = x_sim(2:end) - xi_full;
    control_eff  = diff(x_sim);
    cost_mpc(Np) = sum(tracking_err.^2) + u_flag*sum(control_eff.^2);

    % store deviation trajectory vs clairvoyant
    x_mpc = x_sim(2:end);              % length T_sim
    dev   = x_mpc - x_opt_value;       % MPC - OPT
    TrajDev{Np}     = dev;
    TrajDevMean(Np) = mean(abs(dev));

    % midpoint error around xi: |x_mpc + x_opt - 2*xi|
    mid_err = x_mpc + x_opt_value - 2*xi_full;
    CenterMean(Np) = mean(abs(mid_err));
end

%% PLOT REGRET (as before)
horizons = (1:Np_max)'; 
regret = cost_mpc - cost_opt;
regret = regret(end-size(horizons,1)+1:end);
odd_idx  = mod(horizons,2)==1;
even_idx = ~odd_idx;

% Figure 1: semilogy with odd/even connections
figure(1); clf
semilogy(horizons, regret, '-o','LineWidth',1.5); hold on
semilogy(horizons(odd_idx),  regret(odd_idx),  '-s','LineWidth',1.5);
semilogy(horizons(even_idx), regret(even_idx), '-^','LineWidth',1.5);
xlabel('MPC Horizon N'); ylabel('Exp Regret'); grid on
legend('All','Odd N','Even N','Location','best'); hold off

% Figure 2: loglog with odd/even connections
figure(2); clf
loglog(horizons, regret, '-o','LineWidth',1.5); hold on
loglog(horizons(odd_idx),  regret(odd_idx),  '-s','LineWidth',1.5);
loglog(horizons(even_idx), regret(even_idx), '-^','LineWidth',1.5);
xlabel('MPC Horizon N'); ylabel('LogLog Regret'); grid on
legend('All','Odd N','Even N','Location','best'); hold off

%% 1) trajectories of x_MPC - x_OPT for each horizon
figure(3); clf
t = 1:T_sim;
tiledlayout(5,4);    % 20 tiles, only first 19 used

for Np = 1:Np_max
    nexttile;
    dev = TrajDev{Np};
    plot(t, dev, 'LineWidth', 1.0);
    grid on
    title(sprintf('N = %d', Np));
    if Np > (Np_max-4)
        xlabel('time k');
    end
    if mod(Np-1,4) == 0
        ylabel('x_{MPC} - x_{OPT}');
    end
end

%% 2) "loglog" of mean deviation vs horizon (you currently use semilogy)
figure(4); clf
loglog(horizons, TrajDevMean, 'o-','LineWidth',1.5);
grid on
xlabel('MPC Horizon N');
ylabel('mean |x_{MPC} - x_{OPT}|');
title('Average state deviation vs horizon');

%% 3) loglog of mean |x_MPC + x_OPT - 2*xi| vs horizon
figure(5); clf
loglog(horizons, CenterMean, 'o-','LineWidth',1.5);
grid on
xlabel('MPC Horizon N');
ylabel('mean |x_{MPC} + x_{OPT} - 2\xi|');
title('Average midpoint deviation vs horizon (log-log)');
