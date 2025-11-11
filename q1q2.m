clear; close all;

%% -------------------- PARAMETERS --------------------
T_sim   = 500;          % total sim steps
Np_max  = 30;           % max MPC horizon
x0      = 0;            % initial state
R       = 0.1;          % terminal tube radius
xlim    = 1e1;          % never reached
ulim    = 4/5;          % move limit
xi      = 4/5;          % alternating reference (+/-)
u_flag  = 0;            % set 1 to add move penalty, 0 = constrain only

xi_full = repmat(xi, T_sim, 1);
xi_full(2:2:end) = -xi;

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


%% -------------------- FULL-HORIZON OPT --------------------
x_opt = sdpvar(T_sim,1);
cons_opt = [];
obj_opt  = 0;
xlim_opt = xlim;
for t = 1:T_sim
    obj_opt = obj_opt + (x_opt(t)-xi_full(t))^2;
    if t==1
        obj_opt = obj_opt + u_flag*(x_opt(1)-x0)^2;
        cons_opt = [cons_opt, x_opt(1)-x0 <= ulim, x0 - x_opt(1) <= ulim];
    else
        obj_opt = obj_opt + u_flag*(x_opt(t)-x_opt(t-1))^2;
        cons_opt = [cons_opt, x_opt(t)-x_opt(t-1) <= ulim, x_opt(t-1)-x_opt(t) <= ulim];
    end
    cons_opt = [cons_opt, -xlim_opt <= x_opt(t) <= xlim_opt];
end
opt_sol = optimize(cons_opt, obj_opt, options);
assert(opt_sol.problem==0, 'FULL-OPT failed: %s', yalmiperror(opt_sol.problem));
x_opt_value = value(x_opt);
x_opt_value_ext = [x_opt_value; x_opt_value(end)];  % length T_sim+1
cost_opt    = value(obj_opt);

%% -------------------- BUILD MPC OPTIMIZERS (for every horizon) --------------------
% For each h, build two optimizers sharing the same stage cost/constraints:
%  - DEVIATED terminal: p_term - R <= x_seq(h+1) <= p_term + R
%  - EXACT terminal:    x_seq(h+1) == p_term
F_dev = cell(Np_max,1);
F_exa = cell(Np_max,1);

for h = 1:Np_max
    x_seq = sdpvar(h+1,1);
    p0    = sdpvar(1,1);       % current state
    p_xi  = sdpvar(h,1);       % ref over horizon
    p_term= sdpvar(1,1);       % terminal target (time-shifted OPT)
    
    cons  = [];
    obj   = 0;
    
    % tracking and state bounds for t = 1..h
    for t = 1:h
        obj  = obj + (x_seq(t)-p_xi(t))^2;
        cons = [cons, -xlim <= x_seq(t) <= xlim];
    end
    
    % move limits (and optional penalties) for t = 1..h+1
    for t = 1:h+1
        if t==1
            % move from p0 -> x_seq(1)
            cons = [cons, -ulim <= x_seq(1)-p0 <= ulim];
            obj  = obj + u_flag*(x_seq(1)-p0)^2;
        else
            % move from x_seq(t-1) -> x_seq(t)
            cons = [cons, -ulim <= x_seq(t)-x_seq(t-1) <= ulim];
            obj  = obj + u_flag*(x_seq(t)-x_seq(t-1))^2;
        end
    end
        % Terminal variables x_seq(h+1) are free in stage loop, constrain now:
        cons_dev = [cons, p_term - R <= x_seq(h+1) <= p_term + R];
        cons_exa = [cons, x_seq(h+1) == p_term];
        
        F_dev{h} = optimizer(cons_dev, obj, options, [p0; p_xi; p_term], x_seq);
        F_exa{h} = optimizer(cons_exa, obj, options, [p0; p_xi; p_term], x_seq);
    end

%% -------------------- SIMULATE (apply DEVIATED MPC) & COLLECT DATA --------------------
% We will simulate once for each Np. At each step k we also solve the EXACT terminal
% MPC from the same x(k) and same xi_pred to get the "paired" input.
% For every horizon length k (1..Np_max), we gather samples where Nloc == k.

% Storage per-horizon (cell arrays of samples)
Xnorms   = cell(Np_max,1);   % ||x_{t1}||
DXt2norm = cell(Np_max,1);   % ||x_{t2|t1} - x'_{t2|t1}||
DU_norm  = cell(Np_max,1);   % ||u_dev - u_exa|| = |(x1_dev - x0) - (x1_exa - x0)|

cost_mpc_dev = zeros(Np_max,1);

for Np = 1:Np_max
    x_sim = zeros(T_sim+1,1);
    x_sim(1) = x0;
    
    for k = 1:T_sim
        Nloc = min(Np, T_sim-k+1);
        xi_pred = xi_full(k : k+Nloc-1);
        p_term_k = x_opt_value_ext(k+Nloc);
        
        % Solve both MPCs
        x_pred_dev = F_dev{Nloc}{[ x_sim(k); xi_pred; p_term_k ]};
        x_pred_exa = F_exa{Nloc}{[ x_sim(k); xi_pred; p_term_k ]};
        
        % "Inputs" are first moves: u = x1 - x0
        u_dev = x_pred_dev(1) - x_sim(k);
        u_exa = x_pred_exa(1) - x_sim(k);
        
        % Terminal states (optimal within each MPC subproblem)
        x_term_dev = x_pred_dev(Nloc+1);
        x_term_exa = x_pred_exa(Nloc+1);   % equals p_term_k by construction
        
        % Collect samples only when the active horizon equals the index we’ll bin
        % i.e., data for this Nloc belongs to bin k=Nloc
        if Nloc>=1
            Xnorms{Nloc}(end+1,1)   = abs(x_sim(k));
            DXt2 = abs(x_term_dev - x_term_exa);
            DXt2norm{Nloc}(end+1,1) = DXt2;
            DU_norm{Nloc}(end+1,1)  = abs(u_dev - u_exa);
        end
        
        % Apply DEVIATED policy
        x_sim(k+1) = x_pred_dev(1);
    end
    
    % Total deviated MPC cost for reference
    tracking_err = x_sim(2:end) - xi_full;
    control_eff  = diff(x_sim);
    cost_mpc_dev(Np) = sum(tracking_err.^2) + u_flag*sum(control_eff.^2);
end

%% debug starts
%% -------------------- Per-horizon LS estimate of q2(k) --------------------
% y = ||Δu|| / ||Δx_t2||, LS constant fit per horizon => mean(y)
q2_k = nan(Np_max,1);

for k = 1:Np_max
    DX = DXt2norm{k};
    DU = DU_norm{k};
    if isempty(DX), continue; end

    mask = DX > 1e-8 & isfinite(DU) & isfinite(DX);
    if ~any(mask), continue; end

    y = abs(DU(mask)) ./ abs(DX(mask));
    q2_k(k) = mean(y);                % LS constant fit
end

%% -------------------- POWER-LAW FITS: q2(k) ≈ C * k^{-rho} --------------------
kk = (1:Np_max)';
odd_idx = mod(kk,2)==1;

% ---- choose fit ranges ----
kmin_all = 5;          % <-- change freely
kmin_odd = 5;          % <-- change freely
kmax      = Np_max;    % <-- set < Np_max if you want to truncate the tail

% masks for usable data
valid = q2_k > 0 & isfinite(q2_k) & kk>=1 & kk<=Np_max;

% ----- ALL horizons fit over [kmin_all, kmax] -----
mask_all = valid & kk>=kmin_all & kk<=kmax;
if nnz(mask_all) >= 2
    p_all = polyfit(log(kk(mask_all)), log(q2_k(mask_all)), 1);   % log q2 = m log k + b
    rho2_all = -p_all(1);
    A2_all   = exp(p_all(2));
    q2_fit_all = A2_all * kk.^(-rho2_all);

    % R^2 over the used range
    Y  = log(q2_k(mask_all));  X = log(kk(mask_all));
    Yh = polyval(p_all, X);
    R2_all = 1 - sum((Y-Yh).^2)/sum((Y-mean(Y)).^2);
else
    rho2_all = NaN; A2_all = NaN; q2_fit_all = NaN(size(kk)); R2_all = NaN;
end

% ----- ODD horizons fit over [kmin_odd, kmax] -----
mask_odd = valid & odd_idx & kk>=kmin_odd & kk<=kmax;
have_q2_odd = nnz(mask_odd) >= 2;
if have_q2_odd
    p_odd = polyfit(log(kk(mask_odd)), log(q2_k(mask_odd)), 1);
    rho2_odd = -p_odd(1);
    A2_odd   = exp(p_odd(2));
    q2_fit_odd = A2_odd * kk.^(-rho2_odd);

    Y  = log(q2_k(mask_odd));  X = log(kk(mask_odd));
    Yh = polyval(p_odd, X);
    R2_odd = 1 - sum((Y-Yh).^2)/sum((Y-mean(Y)).^2);
else
    rho2_odd = NaN; A2_odd = NaN; q2_fit_odd = NaN(size(kk)); R2_odd = NaN;
end

%% -------------------- PLOTS --------------------
figure(1); clf
loglog(kk, q2_k, 'o','LineWidth',1.5); hold on
if ~isnan(A2_all)
    loglog(kk, q2_fit_all, '-','LineWidth',1.5);
end
if have_q2_odd
    loglog(kk, q2_fit_odd, '--','LineWidth',1.5);
end
% highlight fitted ranges
plot(kk(mask_all), q2_k(mask_all), 'ko', 'MarkerFaceColor','k');           % used in ALL fit
plot(kk(mask_odd), q2_k(mask_odd), 'ks', 'MarkerFaceColor',[.5 .5 .5]);    % used in ODD fit

grid on; xlabel('horizon k'); ylabel('q_2(k)');
if have_q2_odd
    title(sprintf('q_2 fits: all k≥%d (R^2=%.3f) ~ %.3g k^{-%.3g},  odd k≥%d (R^2=%.3f) ~ %.3g k^{-%.3g}', ...
        kmin_all, R2_all, A2_all, rho2_all, kmin_odd, R2_odd, A2_odd, rho2_odd));
    legend('data','fit: all','fit: odd','ALL fit pts','ODD fit pts','Location','best');
else
    title(sprintf('q_2 fit: all k≥%d (R^2=%.3f) ~ %.3g k^{-%.3g}', ...
        kmin_all, R2_all, A2_all, rho2_all));
    legend('data','fit: all','ALL fit pts','Location','best');
end

%% -------------------- Console summary --------------------
fprintf('q2 LS power-law fits over chosen ranges:\n');
if ~isnan(A2_all)
    fprintf('  all: k in [%d,%d]  ⇒  q2(k) ≈ %.4g * k^{-% .4g}  (R^2=%.3f)\n', ...
        kmin_all, kmax, A2_all, rho2_all, R2_all);
else
    fprintf('  all: insufficient points in the chosen range\n');
end
if have_q2_odd
    fprintf('  odd: k in [%d,%d]  ⇒  q2(k) ≈ %.4g * k^{-% .4g}  (R^2=%.3f)\n', ...
        kmin_odd, kmax, A2_odd, rho2_odd, R2_odd);
else
    fprintf('  odd: insufficient points in the chosen range\n');
end


%% debug ends 

return;


























%% If estimating both q1 and q2
%% -------------------- PER-HORIZON LINEAR FIT : y ≈ a_k * ||x|| + b_k --------------------
% For each k, define y = ||Δu|| / ||Δx_t2|| when ||Δx_t2||>0.
% Then fit [a_k, b_k] by least squares: y ≈ a_k * ||x|| + b_k.
q1_k = nan(Np_max,1);   % a_k
q2_k = nan(Np_max,1);   % b_k

for k = 1:Np_max
    X  = Xnorms{k};
    DX = DXt2norm{k};
    DU = DU_norm{k};
    if isempty(X), continue; end
    mask = DX > 1e-12 & isfinite(DU) & isfinite(X);
    if ~any(mask), continue; end

    x  = X(mask);
    y  = DU(mask)./DX(mask);

    % Linear LS: y ≈ a*x + b
    A = [x, ones(size(x))];
    theta = A \ y;
    ak = theta(1);
    bk = theta(2);

    % Project to nonnegative (optional but sensible for gain bounds)
    q1_k(k) = max(ak,0);
    q2_k(k) = max(bk,0);
end

%% -------------------- POWER-LAW FITS: q_i(k) ≈ C * k^{-rho} --------------------
% Use log-log regression on positive entries.
kk  = (1:Np_max)';
odd_idx  = mod(kk,2)==1;

% q1 fit (all horizons)
mask1 = q1_k > 0;
p1 = polyfit(log(kk(mask1)), log(q1_k(mask1)), 1);   % log q1 = p1(1)*log k + p1(2)
rho1 = -p1(1);
A1   = exp(p1(2));
q1_fit = A1 * kk.^(-rho1);

% q2 fit (all horizons)
mask2_all = q2_k > 0;
p2_all = polyfit(log(kk(mask2_all)), log(q2_k(mask2_all)), 1);
rho2_all = -p2_all(1);
A2_all   = exp(p2_all(2));
q2_fit_all = A2_all * kk.^(-rho2_all);

% q2 fit (odd horizons only)
mask2_odd = q2_k > 0 & odd_idx;
have_q2_odd = nnz(mask2_odd) >= 2;
if have_q2_odd
    p2_odd = polyfit(log(kk(mask2_odd)), log(q2_k(mask2_odd)), 1);
    rho2_odd = -p2_odd(1);
    A2_odd   = exp(p2_odd(2));
    q2_fit_odd = A2_odd * kk.^(-rho2_odd);
else
    rho2_odd = NaN; A2_odd = NaN; q2_fit_odd = NaN(size(kk));
end

%% -------------------- PLOTS --------------------
% 1) q1 and q2 vs horizon
figure(1); clf
subplot(1,2,1)
loglog(kk, q1_k, 'o','LineWidth',1.5); hold on
loglog(kk, q1_fit, '-','LineWidth',1.5);
grid on; xlabel('horizon k'); ylabel('q_1(k)');
title(sprintf('q_1: fit ~ %.3g k^{-%.3g}', A1, rho1));
legend('data','power-law fit','Location','best')

subplot(1,2,2)
loglog(kk, q2_k, 'o','LineWidth',1.5); hold on
loglog(kk, q2_fit_all, '-','LineWidth',1.5);
if have_q2_odd
    loglog(kk, q2_fit_odd, '--','LineWidth',1.5);
    legend_entries = {'data','fit: all k','fit: odd k'};
else
    legend_entries = {'data','fit: all k'};
end
grid on; xlabel('horizon k'); ylabel('q_2(k)');
if have_q2_odd
    title(sprintf('q_2: all ~ %.3g k^{-%.3g}, odd ~ %.3g k^{-%.3g}', ...
        A2_all, rho2_all, A2_odd, rho2_odd));
else
    title(sprintf('q_2: all ~ %.3g k^{-%.3g}', A2_all, rho2_all));
end
legend(legend_entries,'Location','best')

% 2) Optional: show regret of the applied (deviated) MPC vs OPT
figure(2); clf
regret = cost_mpc_dev - cost_opt;
semilogy(kk, regret, '-o','LineWidth',1.5); grid on
xlabel('MPC Horizon k'); ylabel('Regret (deviated MPC minus OPT)');
title('Regret vs horizon (deviated terminal)');

%% -------------------- Console summary --------------------
fprintf('Power-law fits (log-log LS):\n');
fprintf('  q1(k) ≈ %.4g * k^{-% .4g}\n', A1, rho1);
fprintf('  q2 all(k) ≈ %.4g * k^{-% .4g}\n', A2_all, rho2_all);
if have_q2_odd
    fprintf('  q2 odd(k) ≈ %.4g * k^{-% .4g}\n', A2_odd, rho2_odd);
else
    fprintf('  q2 odd(k): insufficient positive odd points to fit\n');
end
