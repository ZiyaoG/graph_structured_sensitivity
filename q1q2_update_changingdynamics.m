clear; close all;

%% -------------------- PARAMETERS --------------------
a_sys  = 1.0;        % system parameter a in x_{t+1} = a x_t + b u_t
b_sys  = 1.0;        % system parameter b (assume ~= 0)

T_sim   = 500;                          % total sim steps
Np_list = [3 4 5 6 7 8 9 10 11 15 20 21 31 51 71];
Np_max  = max(Np_list);

x0      = 0;            % initial state
R       = 0.1;          % terminal tube radius
xlim    = 1e0;          % state limit, never reached in practice
ulim    = 4/5;          % input limit |u_t| <= ulim
xi      = 4/5;          % alternating reference (+/- each step)
u_flag  = 0;            % 1 adds input penalty, 0 no input penalty

% Build alternating reference long enough to support max horizon lookahead
xi_full_ext = repmat(xi, T_sim + Np_max, 1);
xi_full_ext(2:2:end) = -xi;

% YALMIP and Gurobi options
options = sdpsettings('solver','gurobi','verbose',0,'cachesolvers',1);
options.gurobi.Threads        = 1;
options.gurobi.Method         = 2;
options.gurobi.Crossover      = 0;
options.gurobi.NumericFocus   = 3;
options.gurobi.ScaleFlag      = 2;
options.gurobi.Presolve       = 1;
options.gurobi.FeasibilityTol = 1e-9;
options.gurobi.OptimalityTol  = 1e-9;
options.gurobi.BarConvTol     = 1e-12;
options.gurobi.TuneTimeLimit  = 0;

%% -------------------- FULL-HORIZON OPT --------------------
% Clairvoyant controller for comparison, dynamics x_{t+1} = a x_t + b u_t
x_opt = sdpvar(T_sim,1);   % states x_1,...,x_T
u_opt = sdpvar(T_sim,1);   % inputs u_0,...,u_{T-1}

cons_opt = [];
obj_opt  = 0;
for t = 1:T_sim
    % tracking cost
    obj_opt = obj_opt + (x_opt(t) - xi_full_ext(t))^2;
    % optional input penalty
    obj_opt = obj_opt + u_flag * u_opt(t)^2;

    % state and input bounds
    cons_opt = [cons_opt, -xlim <= x_opt(t) <= xlim];
    cons_opt = [cons_opt, -ulim <= u_opt(t) <= ulim];

    % dynamics
    if t == 1
        cons_opt = [cons_opt, x_opt(1) == a_sys * x0 + b_sys * u_opt(1)];
    else
        cons_opt = [cons_opt, x_opt(t) == a_sys * x_opt(t-1) + b_sys * u_opt(t)];
    end
end

opt_sol = optimize(cons_opt, obj_opt, options);
assert(opt_sol.problem == 0, 'FULL-OPT failed: %s', yalmiperror(opt_sol.problem));
x_opt_value      = value(x_opt);
x_opt_value_ext  = [x_opt_value; x_opt_value(end)];  % length T_sim+1
cost_opt         = value(obj_opt); %#ok<NASGU>

%% -------------------- BUILD MPC OPTIMIZERS --------------------
% Now build MPC problems with dynamics x_{t+1} = a x_t + b u_t
F_dev = containers.Map('KeyType','double','ValueType','any');
F_exa = containers.Map('KeyType','double','ValueType','any');

for h = Np_list(:)'
    x_seq = sdpvar(h+1,1);   % states x_{k+1},...,x_{k+h+1}
    u_seq = sdpvar(h+1,1);   % inputs u_k,...,u_{k+h}
    p0    = sdpvar(1,1);     % current state x_k
    p_xi  = sdpvar(h,1);     % reference over horizon (for x_{k+1}..x_{k+h})
    p_term= sdpvar(1,1);     % terminal target for x_{k+h}

    cons  = [];
    obj   = 0;

    % stage cost and bounds for t = 1..h (states x_{k+1}..x_{k+h})
    for t = 1:h
        obj = obj + (x_seq(t) - p_xi(t))^2;
        obj = obj + u_flag * u_seq(t)^2;

        cons = [cons, -xlim <= x_seq(t) <= xlim];
        cons = [cons, -ulim <= u_seq(t) <= ulim];
    end

    % bounds and optional penalty for last state and input (t = h+1)
    cons = [cons, -xlim <= x_seq(h+1) <= xlim];
    cons = [cons, -ulim <= u_seq(h+1) <= ulim];
    obj  = obj + u_flag * u_seq(h+1)^2;

    % dynamics: x_{k+1}..x_{k+h+1}
    cons = [cons, x_seq(1) == a_sys * p0 + b_sys * u_seq(1)];
    for t = 2:h+1
        cons = [cons, x_seq(t) == a_sys * x_seq(t-1) + b_sys * u_seq(t)];
    end

    % terminal constraints
    cons_dev = [cons, p_term - R <= x_seq(h+1) <= p_term + R];
    cons_exa = [cons, x_seq(h+1) == p_term];

    F_dev(h) = optimizer(cons_dev, obj, options, [p0; p_xi; p_term], x_seq);
    F_exa(h) = optimizer(cons_exa, obj, options, [p0; p_xi; p_term], x_seq);
end

%% -------------------- SIMULATE (apply DEVIATED MPC) --------------------
Xnorms       = cell(Np_max,1);
DXt2norm     = cell(Np_max,1);
DU_norm      = cell(Np_max,1);
cost_mpc_dev = NaN(Np_max,1);

for Np = Np_list(:)'
    x_sim = zeros(T_sim+1,1);
    x_sim(1) = x0;

    for k = 1:T_sim
        h = Np;

        % predicted reference for states x_{k+1}..x_{k+h}
        xi_pred  = xi_full_ext(k : k+h-1);

        % terminal target index in clairvoyant trajectory
        term_idx = min(k+h, T_sim+1);
        p_term_k = x_opt_value_ext(term_idx);

        % solve both MPC problems
        Fh_dev = F_dev(h);
        Fh_exa = F_exa(h);
        x_pred_dev = Fh_dev{[ x_sim(k); xi_pred; p_term_k ]};
        x_pred_exa = Fh_exa{[ x_sim(k); xi_pred; p_term_k ]};

        % first control input (consistent with dynamics)
        u_dev = (x_pred_dev(1) - a_sys * x_sim(k)) / b_sys;
        u_exa = (x_pred_exa(1) - a_sys * x_sim(k)) / b_sys;

        % terminal states inside subproblems
        x_term_dev = x_pred_dev(h+1);
        x_term_exa = x_pred_exa(h+1);

        % collect samples for horizon h
        Xnorms{h}(end+1,1)   = abs(x_sim(k));
        DXt2norm{h}(end+1,1) = abs(x_term_dev - x_term_exa);
        DU_norm{h}(end+1,1)  = abs(u_dev - u_exa);

        % apply deviated policy to the true system
        x_sim(k+1) = a_sys * x_sim(k) + b_sys * u_dev;
    end

    % MPC cost for this horizon
    tracking_err = x_sim(2:end) - xi_full_ext(1:T_sim);
    u_seq_sim    = (x_sim(2:end) - a_sys * x_sim(1:end-1)) / b_sys;
    cost_mpc_dev(Np) = sum(tracking_err.^2) + u_flag * sum(u_seq_sim.^2);
end

%% -------------------- Per-horizon LS estimate of q2(k) --------------------
q2_k = NaN(Np_max,1);
for k = Np_list(:)'
    DX = DXt2norm{k};
    DU = DU_norm{k};
    if isempty(DX), continue; end
    mask = DX > 1e-8 & isfinite(DU) & isfinite(DX);
    if ~any(mask), continue; end
    y = abs(DU(mask)) ./ abs(DX(mask));
    q2_k(k) = mean(y);
end

%% -------------------- POWER-LAW FITS: q2(k) ≈ C * k^{-rho} --------------------
kk      = Np_list(:);
q2_sel  = q2_k(kk);
odd_idx = mod(kk,2)==1;

kmin_all = 5;
kmin_odd = kmin_all;
kmax     = max(kk);

valid = q2_sel > 0 & isfinite(q2_sel);

mask_odd   = valid & odd_idx & kk>=kmin_odd & kk<=kmax;
have_q2_odd = nnz(mask_odd) >= 2;
if have_q2_odd
    p_odd      = polyfit(log(kk(mask_odd)), log(q2_sel(mask_odd)), 1);
    rho2_odd   = -p_odd(1);
    A2_odd     = exp(p_odd(2));
    q2_fit_odd = A2_odd * kk.^(-rho2_odd);

    Y  = log(q2_sel(mask_odd));  X = log(kk(mask_odd));
    Yh = polyval(p_odd, X);
    R2_odd = 1 - sum((Y-Yh).^2)/sum((Y-mean(Y)).^2);
else
    rho2_odd = NaN; A2_odd = NaN; q2_fit_odd = NaN(size(kk)); R2_odd = NaN;
end

%% -------------------- PLOTS --------------------
figure(1); clf
loglog(kk, q2_sel, 'o','LineWidth',1.5); hold on
if have_q2_odd
    loglog(kk, q2_fit_odd, '--','LineWidth',1.5);
    plot(kk(mask_odd), q2_sel(mask_odd), 'ks', 'MarkerFaceColor',[.5 .5 .5]);
end

grid on; xlabel('horizon k'); ylabel('q_2(k)');
if have_q2_odd
    title(sprintf('q_2 fits:  odd k≥%d (R^2=%.3f) ~ %.3g k^{-%.3g}', ...
        kmin_odd, R2_odd, A2_odd, rho2_odd));
    legend('data','fit: odd','ODD fit pts','Location','best');
end

%% --- New figure: linear x, log y ---
figure(2); clf
semilogy(kk, q2_sel, 'o', 'LineWidth', 1.5); hold on
if have_q2_odd
    semilogy(kk(mask_odd), q2_sel(mask_odd), 'ks', 'MarkerFaceColor', [0.5 0.5 0.5]);
end

grid on
xlabel('horizon k');
ylabel('q_2(k)');
title('q_2 vs horizon, y-axis log scale');
legend('data','fit: odd','ODD fit pts','Location','best');


%% -------------------- Console summary --------------------
fprintf('q2 LS power-law fits over chosen ranges (Np_list = [%s])\n', num2str(kk'));
if have_q2_odd
    fprintf('  odd: k in [%d,%d]  =>  q2(k) ≈ %.4g * k^{-% .4g}  (R^2=%.3f)\n', ...
        max(kmin_odd,min(kk(odd_idx))), max(kk(odd_idx)), A2_odd, rho2_odd, R2_odd);
else
    fprintf('  odd: insufficient points in the chosen range\n');
end
