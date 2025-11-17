clear; close all;

%% PARAMETERS
T_sim    = 500;                   % Total sim steps
x0       = 0;                     % Initial state
R        = 0.01;                   % terminal tube radius
xlim     = 1e4;                   % state bounds (effectively inactive)
ulim     = 4/5;                   % move limit
u_flag   = 0;                     % 1: effort penalty, 0: none

% Horizons you control:
Np_list  = 3:1:29;                % example: [2 5 8 ... 32]
Np_max   = max(Np_list);
num_h    = numel(Np_list);

% Number of random reference trajectories to average over
num_seeds = 1;

% Random-walk reference parameters
step_mag_factor = 2;              % step magnitude relative to ulim (>1)
step_mag        = step_mag_factor * ulim;

%% SOLVER OPTIONS
options = sdpsettings('solver','gurobi','verbose',0);

%% PRE-BUILD MPC OPTIMIZERS FOR ALL HORIZONS h = 1..Np_max
% Decision: x_seq(0..h) -> x_seq(1..h+1)
% Params: [p0; p_xi; p_term]
% Terminal: p_term - R <= x_seq(h+1) <= p_term + R
F_all = cell(Np_max,1);

for h = 1:Np_max
    x_seq  = sdpvar(h+1,1);   % x(0..h)
    p0     = sdpvar(1,1);     % current state
    p_xi   = sdpvar(h,1);     % reference over horizon
    p_term = sdpvar(1,1);     % terminal tube center

    cons = [];
    obj  = 0;

    for t = 1:h
        obj = obj + (x_seq(t) - p_xi(t))^2;  % tracking cost

        if t==1
            obj  = obj + u_flag * (x_seq(1) - p0)^2;
            cons = [cons, x_seq(1) - p0 <= ulim, p0 - x_seq(1) <= ulim];
        else
            obj  = obj + u_flag * (x_seq(t) - x_seq(t-1))^2;
            cons = [cons, x_seq(t) - x_seq(t-1) <= ulim, ...
                           x_seq(t-1) - x_seq(t) <= ulim];
        end

        cons = [cons, -xlim <= x_seq(t) <= xlim];
    end

    % couple final state to previous via move limit
    obj  = obj + u_flag * (x_seq(h+1) - x_seq(h))^2;
    cons = [cons, x_seq(h+1) - x_seq(h) <= ulim, ...
                   x_seq(h)   - x_seq(h+1) <= ulim];

    % terminal tube around p_term
    cons = [cons, p_term - R <= x_seq(h+1) <= p_term + R];

    F_all{h} = optimizer(cons, obj, options, [p0; p_xi; p_term], x_seq);
end

%% STORAGE OVER SEEDS
cost_opt_all    = zeros(num_seeds,1);          % clairvoyant cost per seed
cost_mpc_all    = zeros(num_seeds, num_h);     % MPC cost per seed & horizon
TrajDevMean_all = zeros(num_seeds, num_h);     % mean |x_MPC - x_OPT|
CenterMean_all  = zeros(num_seeds, num_h);     % mean |x_MPC + x_OPT - 2*xi|

% sample trajectories x_MPC - x_OPT for the first seed only
TrajDev_sample  = cell(num_h,1);

%% MAIN LOOP OVER SEEDS
for s = 1:num_seeds
    %% BUILD RANDOM-WALK REFERENCE FOR THIS SEED
    rng(s);                         % different reference per seed

    xi_full = zeros(T_sim,1);
    xi_full(1) = step_mag * sign(randn);   % initial sign random
    if xi_full(1) == 0
        xi_full(1) = step_mag;
    end

    for t = 2:T_sim
        xi_full(t) = xi_full(t-1) + step_mag * sign(randn);
    end

    %% FULL-HORIZON OPT (CLAIRVOYANT) FOR THIS REFERENCE
    x_opt    = sdpvar(T_sim,1);
    cons_opt = [];
    obj_opt  = 0;

    for t = 1:T_sim
        obj_opt = obj_opt + (x_opt(t) - xi_full(t))^2;

        if t==1
            obj_opt = obj_opt + u_flag * (x_opt(1) - x0)^2;
            cons_opt = [cons_opt, x_opt(1)-x0 <= ulim, x0 - x_opt(1) <= ulim];
        else
            obj_opt = obj_opt + u_flag * (x_opt(t) - x_opt(t-1))^2;
            cons_opt = [cons_opt, x_opt(t) - x_opt(t-1) <= ulim, ...
                                 x_opt(t-1) - x_opt(t) <= ulim];
        end

        cons_opt = [cons_opt, -xlim <= x_opt(t) <= xlim];
    end

    opt_sol = optimize(cons_opt, obj_opt, options);
    if opt_sol.problem
        error('FULL-OPT failed (seed %d): %s', s, yalmiperror(opt_sol.problem));
    end

    cost_opt    = value(obj_opt);
    x_opt_value = value(x_opt);

    cost_opt_all(s) = cost_opt;

    %% MPC SIMULATIONS FOR ALL HORIZONS IN Np_list
    for idxH = 1:num_h
        Np = Np_list(idxH);

        x_sim = zeros(T_sim+1,1);
        x_sim(1) = x0;

        for k = 1:T_sim
            % local horizon near end
            Nloc = min(Np, T_sim - k + 1);

            % pick optimizer
            F = F_all{Nloc};

            % local reference slice
            xi_pred = xi_full(k : k+Nloc-1);

            % terminal center = clairvoyant x_opt at time k+Nloc
            term_idx = min(k + Nloc, T_sim);
            p_term_k = x_opt_value(term_idx);

            % solve MPC subproblem
            x_pred = F{[ x_sim(k); xi_pred; p_term_k ]};

            % apply first move
            x_sim(k+1) = x_pred(1);
        end

        % MPC cost for this horizon & seed
        x_mpc = x_sim(2:end);              % length T_sim
        tracking_err = x_mpc - xi_full;
        control_eff  = diff(x_sim);
        cost_mpc     = sum(tracking_err.^2) + u_flag * sum(control_eff.^2);

        cost_mpc_all(s, idxH) = cost_mpc;

        % deviations vs clairvoyant
        dev = x_mpc - x_opt_value;
        TrajDevMean_all(s, idxH) = mean(abs(dev));

        % midpoint error |x_MPC + x_OPT - 2*xi|
        mid_err = x_mpc + x_opt_value - 2*xi_full;
        CenterMean_all(s, idxH) = mean(abs(mid_err));

        % store sample trajectories for seed 1
        if s == 1
            TrajDev_sample{idxH} = dev;
        end
    end
end

%% AVERAGE OVER SEEDS
regret_all   = cost_mpc_all - cost_opt_all;   % implicit expansion in recent MATLAB
regret_mean  = mean(regret_all, 1);           % 1 x num_h
TrajDevMean  = mean(TrajDevMean_all, 1);
CenterMean   = mean(CenterMean_all, 1);

horizons = Np_list(:);
odd_idx  = mod(horizons,2)==1;
even_idx = ~odd_idx;

%% Figure 1: semilogy of average regret vs horizon
figure(1); clf
semilogy(horizons, regret_mean, '-o','LineWidth',1.5); hold on
semilogy(horizons(odd_idx),  regret_mean(odd_idx),  's','LineWidth',1.0);
semilogy(horizons(even_idx), regret_mean(even_idx), '^','LineWidth',1.0);
xlabel('MPC Horizon N'); ylabel('Average regret over seeds'); grid on
legend('Mean regret','Odd N','Even N','Location','best');
title(sprintf('Average regret over %d random references', num_seeds));
hold off

%% Figure 2: loglog of average regret vs horizon
figure(2); clf
loglog(horizons, regret_mean, '-o','LineWidth',1.5); hold on
loglog(horizons(odd_idx),  regret_mean(odd_idx),  's','LineWidth',1.0);
loglog(horizons(even_idx), regret_mean(even_idx), '^','LineWidth',1.0);
xlabel('MPC Horizon N'); ylabel('Average regret over seeds'); grid on
legend('Mean regret','Odd N','Even N','Location','best');
title(sprintf('Average regret (log-log) over %d random references', num_seeds));
hold off

%% Figure 3: sample trajectories of x_MPC - x_OPT (seed 1) for each horizon
figure(3); clf
t = 1:T_sim;
rows = 4; cols = 3;          % 12 tiles, we use first num_h = 11
tiledlayout(rows, cols);

for idxH = 1:num_h
    nexttile;
    dev = TrajDev_sample{idxH};
    plot(t, dev, 'LineWidth', 1.0);
    grid on
    title(sprintf('N = %d', horizons(idxH)));
    if idxH > num_h - cols
        xlabel('time k');
    end
    if mod(idxH-1, cols) == 0
        ylabel('x_{MPC} - x_{OPT}');
    end
end

%% Figure 4: semilogy of average mean |x_MPC - x_OPT| vs horizon
figure(4); clf
semilogy(horizons, TrajDevMean, 'o-','LineWidth',1.5);
grid on
xlabel('MPC Horizon N');
ylabel('mean |x_{MPC} - x_{OPT}| (averaged over seeds)');
title('Average state deviation vs horizon');

%% Figure 5: loglog of average mean |x_MPC + x_OPT - 2\xi| vs horizon
figure(5); clf
loglog(horizons, CenterMean, 'o-','LineWidth',1.5);
grid on
xlabel('MPC Horizon N');
ylabel('mean |x_{MPC} + x_{OPT} - 2\xi| (averaged over seeds)');
title('Average midpoint deviation vs horizon (log-log)');
