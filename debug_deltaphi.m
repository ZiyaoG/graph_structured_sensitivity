clear; close all;

%% -------------------- PARAMETERS --------------------
T_sim   = 500;                          % total sim steps
Np_list = [1 2 3 4 5 6 7 8 9 10 11 15 20 21 31 51 ];   % <-- choose any positive integers
Np_max  = max(Np_list);

x0      = 0;            % initial state
R       = 0.1;          % terminal tube radius
xlim    = 1e4;          % never reached in practice
ulim    = 4/5;          % move limit
xi      = 4/5;          % alternating reference (+/- each step)
u_flag  = 0;            % 1 adds move penalty, 0 constrain only

% Build alternating reference long enough to support max horizon lookahead
xi_full_ext = repmat(xi, T_sim + Np_max, 1);
xi_full_ext(2:2:end) = -xi;
% Remark: to make u consrtraint active at every step, you either make the referecne
% has jumping signs, or very big. anyway, you want to make it never
% reached, such that u constraint is always active. This is the way to
% achieve polynomial regret. 1/N or 1/N^2 (if 1/N cancceled)
% xi_full = repmat(xi, T_sim, 1);
% xi_full(2:2:end) = -xi;
T_ext = T_sim + Np_max;
r = 1.2 * ulim;                  % slope > u_max
t_vec = (1:T_ext)';              
xi_full = r * t_vec;             % pure ramp

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
options.gurobi.TuneTimeLimit  = 0;      % fixes "value -1" error

%% -------------------- FULL-HORIZON OPT --------------------
x_opt = sdpvar(T_sim,1);
cons_opt = [];
obj_opt  = 0;
for t = 1:T_sim
    obj_opt = obj_opt + (x_opt(t)-xi_full_ext(t))^2;
    if t==1
        obj_opt = obj_opt + u_flag*(x_opt(1)-x0)^2;
        cons_opt = [cons_opt, x_opt(1)-x0 <= ulim, x0 - x_opt(1) <= ulim];
    else
        obj_opt = obj_opt + u_flag*(x_opt(t)-x_opt(t-1))^2;
        cons_opt = [cons_opt, x_opt(t)-x_opt(t-1) <= ulim, x_opt(t-1)-x_opt(t) <= ulim];
    end
    cons_opt = [cons_opt, -xlim <= x_opt(t) <= xlim];
end
opt_sol = optimize(cons_opt, obj_opt, options);
assert(opt_sol.problem==0, 'FULL-OPT failed: %s', yalmiperror(opt_sol.problem));
x_opt_value = value(x_opt);
x_opt_value_ext = [x_opt_value; x_opt_value(end)];  % length T_sim+1
cost_opt    = value(obj_opt); %#ok<NASGU>

%% -------------------- BUILD MPC OPTIMIZERS (only horizons in Np_list) --------------------
% For each h in Np_list, build two optimizers sharing stage structure:
%   - DEVIATED terminal: p_term - R <= x_seq(h+1) <= p_term + R
%   - EXACT terminal:    x_seq(h+1) == p_term
F_dev = containers.Map('KeyType','double','ValueType','any');
F_exa = containers.Map('KeyType','double','ValueType','any');

for h = Np_list(:)'
    x_seq = sdpvar(h+1,1);
    p0    = sdpvar(1,1);       % current state
    p_xi  = sdpvar(h,1);       % ref over horizon
    p_term= sdpvar(1,1);       % terminal target

    cons  = [];
    obj   = 0;

    % tracking and state bounds for t = 1..h
    for t = 1:h
        obj  = obj + (x_seq(t)-p_xi(t))^2;
        cons = [cons, -xlim <= x_seq(t) <= xlim];
    end

    % move limits and optional penalties for t = 1..h+1
    for t = 1:h+1
        if t==1
            cons = [cons, -ulim <= x_seq(1)-p0 <= ulim];
            obj  = obj + u_flag*(x_seq(1)-p0)^2;
        else
            cons = [cons, -ulim <= x_seq(t)-x_seq(t-1) <= ulim];
            obj  = obj + u_flag*(x_seq(t)-x_seq(t-1))^2;
        end
    end

    % terminal constraints
    cons_dev = [cons, p_term - R <= x_seq(h+1) <= p_term + R];
    cons_exa = [cons, x_seq(h+1) == p_term];

    F_dev(h) = optimizer(cons_dev, obj, options, [p0; p_xi; p_term], x_seq);
    F_exa(h) = optimizer(cons_exa, obj, options, [p0; p_xi; p_term], x_seq);
end

%% -------------------- SIMULATE (apply DEVIATED MPC) & COLLECT DATA --------------------
% For each selected horizon Np, always solve with that horizon, no clipping.
Xnorms       = cell(Np_max,1);    % bin by horizon, index with k=h
DXt2norm     = cell(Np_max,1);
DU_norm      = cell(Np_max,1);
TermDev      = cell(Np_max,1);    % NEW: terminal deviation |x_seq(h+1) - p_term|
cost_mpc_dev = NaN(Np_max,1);     % unused horizons remain NaN

for Np = Np_list(:)'
    x_sim = zeros(T_sim+1,1);
    x_sim(1) = x0;

    for k = 1:T_sim
        h = Np;  % fixed horizon
        xi_pred  = xi_full_ext(k : k+h-1);
        % clamp terminal index to the last available full-opt point
        term_idx = min(k+h, T_sim+1);
        p_term_k = x_opt_value_ext(term_idx);

        % Solve both MPCs of horizon h
        Fh_dev = F_dev(h);                 % fetch optimizer object
        Fh_exa = F_exa(h);
        x_pred_dev = Fh_dev{[ x_sim(k); xi_pred; p_term_k ]};
        x_pred_exa = Fh_exa{[ x_sim(k); xi_pred; p_term_k ]};

        % first move as input (here: position difference acts like "u")
        u_dev = x_pred_dev(1) - x_sim(k);
        u_exa = x_pred_exa(1) - x_sim(k);

        % terminal states within subproblems
        x_term_dev = x_pred_dev(h+1);
        x_term_exa = x_pred_exa(h+1);

        % collect samples into bin h
        Xnorms{h}(end+1,1)   = abs(x_sim(k));
        DXt2norm{h}(end+1,1) = abs(x_term_dev - x_term_exa);
        DU_norm{h}(end+1,1)  = abs(u_dev - u_exa);

        % NEW: store terminal deviation from desired terminal state p_term_k
        TermDev{h}(end+1,1) = abs(x_term_dev - p_term_k);

        % apply DEVIATED policy
        x_sim(k+1) = x_pred_dev(1);
    end

    % total deviated MPC cost for this horizon
    tracking_err = x_sim(2:end) - xi_full_ext(1:T_sim);
    control_eff  = diff(x_sim);
    cost_mpc_dev(Np) = sum(tracking_err.^2) + u_flag*sum(control_eff.^2);
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
    q2_k(k) = mean(y);     % LS constant fit
end

%% -------------------- POWER-LAW FITS: q2(k) ≈ C * k^{-rho} --------------------
kk = Np_list(:);                  % only selected horizons
q2_sel = q2_k(kk);
odd_idx = mod(kk,2)==1;

% choose fit ranges
kmin_all = 31;      % you can change
kmin_odd = kmin_all;
kmax     = max(kk);

valid = q2_sel > 0 & isfinite(q2_sel);

% % ALL horizons fit (kept commented as before)
% mask_all = valid & kk>=kmin_all & kk<=kmax;
% if nnz(mask_all) >= 2
%     p_all = polyfit(log(kk(mask_all)), log(q2_sel(mask_all)), 1);
%     rho2_all = -p_all(1);
%     A2_all   = exp(p_all(2));
%     q2_fit_all = A2_all * kk.^(-rho2_all);
%
%     Y  = log(q2_sel(mask_all));  X = log(kk(mask_all));
%     Yh = polyval(p_all, X);
%     R2_all = 1 - sum((Y-Yh).^2)/sum((Y-mean(Y)).^2);
% else
%     rho2_all = NaN; A2_all = NaN; q2_fit_all = NaN(size(kk)); R2_all = NaN;
% end

% ODD horizons fit
mask_odd = valid & odd_idx & kk>=kmin_odd & kk<=kmax;
have_q2_odd = nnz(mask_odd) >= 2;
if have_q2_odd
    p_odd = polyfit(log(kk(mask_odd)), log(q2_sel(mask_odd)), 1);
    rho2_odd = -p_odd(1);
    A2_odd   = exp(p_odd(2));
    q2_fit_odd = A2_odd * kk.^(-rho2_odd);

    Y  = log(q2_sel(mask_odd));  X = log(kk(mask_odd));
    Yh = polyval(p_odd, X);
    R2_odd = 1 - sum((Y-Yh).^2)/sum((Y-mean(Y)).^2);
else
    rho2_odd = NaN; A2_odd = NaN; q2_fit_odd = NaN(size(kk)); R2_odd = NaN;
end

%% -------------------- PLOTS: q2(k) --------------------
figure(1); clf
loglog(kk, q2_sel, 'o','LineWidth',1.5); hold on
% if ~isnan(A2_all), loglog(kk, q2_fit_all, '-','LineWidth',1.5); end
if have_q2_odd,   loglog(kk, q2_fit_odd, '--','LineWidth',1.5); end

if have_q2_odd
    plot(kk(mask_odd), q2_sel(mask_odd), 'ks', 'MarkerFaceColor',[.5 .5 .5]);
end

grid on; xlabel('horizon k'); ylabel('q_2(k)');
if have_q2_odd
    title(sprintf('q_2 fits:  odd k≥%d (R^2=%.3f) ~ %.3g k^{-%.3g}', ...
        kmin_odd, R2_odd, A2_odd, rho2_odd));
    legend('data','fit: odd','ODD fit pts','Location','best');
else
    title(sprintf('q_2 fit: all k≥%d (R^2=%.3f) ~ %.3g k^{-%.3g}', ...
        kmin_all, R2_all, A2_all, R2_all));
    legend('data','fit: all','ALL fit pts','Location','best');
end

%% -------------------- Terminal deviation vs horizon --------------------
termDev_mean = NaN(Np_max,1);
termDev_max  = NaN(Np_max,1);

for k = Np_list(:)'
    d = TermDev{k};
    if isempty(d), continue; end
    termDev_mean(k) = mean(d);
    termDev_max(k)  = max(d);
end

figure(2); clf
loglog(kk, termDev_mean(kk), 'o-','LineWidth',1.5); hold on
loglog(kk, termDev_max(kk),  's--','LineWidth',1.5);
grid on
xlabel('horizon k');
ylabel('|x_{h+1} - p_{term}|');
title('Terminal-state deviation vs horizon');
legend('mean deviation','max deviation','Location','best');

%% -------------------- Console summary --------------------
fprintf('q2 LS power-law fits over chosen ranges (Np_list = [%s])\n', num2str(kk'));
if have_q2_odd
    fprintf('  odd: k in [%d,%d]  =>  q2(k) ≈ %.4g * k^{-% .4g}  (R^2=%.3f)\n', ...
        max(kmin_odd,min(kk(odd_idx))), max(kk(odd_idx)), A2_odd, rho2_odd, R2_odd);
else
    fprintf('  odd: insufficient points in the chosen range\n');
end
