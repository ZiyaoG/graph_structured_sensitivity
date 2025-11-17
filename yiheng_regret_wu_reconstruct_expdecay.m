clear; close all;

%% PARAMETERS
T_sim   = 500;         % Total sim steps
Np_max  = 21;          % Max MPC horizon to test
x0      = 0;           % Initial state
R       = 0.1;
% xi_full = repmat(4/5, T_sim, 1);
% xi_full(2:2:end) = -4/5;
xlim = 1e1;   % never reached
ulim = 4/5;
xi = 4/5;
xi_full = repmat(xi, T_sim, 1);
xi_full(2:2:end) = -xi;

u_flag=0;

options = sdpsettings('solver','gurobi','verbose',0);
% options = sdpsettings('solver','gurobi','verbose',0, ...
%     'gurobi.Method', 1);   % 0=primal simplex, 1=dual simplex, 2=barrier    


%% FULL-HORIZON OPT 
% options = sdpsettings('solver','gurobi','debug',0,'verbose',0); % for debugging
options = sdpsettings('solver','gurobi','verbose',0);

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
% obj_opt  = obj_opt + x_opt(T_sim+1)^2 + (x_opt(T_sim+1)-x_opt(T_sim))^2;
opt_sol = optimize(cons_opt, obj_opt, options);
if opt_sol.problem
    error('FULL-OPT failed with Gurobi: %s', yalmiperror(opt_sol.problem));
end
cost_opt = value(obj_opt);

%% debug - to study the behavior of full horizon optimization
% plot(1:T_sim-1, abs(diff(value(x_opt))),'-o')
x_opt_value=value(x_opt);

% return;

%% PRE-BUILD OPTIMIZERS FOR ALL HORIZONS h=1..Np_max
F_all = cell(Np_max,1);
for h = 1:Np_max
    x_seq = sdpvar(h+1,1);
    p0    = sdpvar(1,1);
    p_xi  = sdpvar(h,1);
    cons  = [];
    obj   = 0;

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

    % --- NEW: couple the terminal state (t = h+1) to x(h) via move limit
    obj  = obj + u_flag*(x_seq(h+1)-x_seq(h))^2;    % optional effort on last move
    cons = [cons, x_seq(h+1)-x_seq(h) <= ulim, x_seq(h)-x_seq(h+1) <= ulim];

    cons = [cons, x_opt_value(h+1)-R<=x_seq(h+1) <= x_opt_value(h+1)+R];
    F_all{h} = optimizer(cons, obj, options, [p0; p_xi], x_seq);
end


%% MPC SIMULATIONS FOR Np = 1..Np_max
cost_mpc = zeros(Np_max,1);
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
        
        % solve
        x_pred = F{[ x_sim(k); xi_pred ]};
        
        % apply first move
        x_sim(k+1) = x_pred(1);
    end
    
    % compute total cost
    tracking_err = x_sim(2:end) - xi_full;
    control_eff  = diff(x_sim);
    cost_mpc(Np) = sum(tracking_err.^2) + u_flag*sum(control_eff.^2);
end


%% PLOT REGRET
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


%
% --- Annotate slopes if R^2 > 0.98 (no fit lines drawn) ---
R2 = 0.99;
odd_idx  = mod(horizons,2)==1;
even_idx = ~odd_idx;

% Figure 1 (semilogy): fit log10(regret) ~ m * horizon + b
figure(1); hold on
% Odd subset
x = horizons(odd_idx);  y = regret(odd_idx);
mask = (y > 0);  x = x(mask);  y = y(mask);
if numel(x) >= 2
    X = x;              Y = log10(y);
    p = polyfit(X, Y, 1);           m = p(1);
    Yhat = polyval(p, X);
    r2 = 1 - sum((Y - Yhat).^2) / sum((Y - mean(Y)).^2);
    if r2 > R2
        xlab = x(end-1) + 0.02*(max(horizons)-min(horizons));
        ylab = y(end-1);
        text(xlab, ylab, sprintf(' %.3g', m), ...
            'HorizontalAlignment','left','VerticalAlignment','middle');
    end
end
% Even subset
x = horizons(even_idx);  y = regret(even_idx);
mask = (y > 0);  x = x(mask);  y = y(mask);
if numel(x) >= 2
    X = x;              Y = log10(y);
    p = polyfit(X, Y, 1);           m = p(1);
    Yhat = polyval(p, X);
    r2 = 1 - sum((Y - Yhat).^2) / sum((Y - mean(Y)).^2);
    if r2 > R2
        xlab = x(end-1) + 0.02*(max(horizons)-min(horizons));
        ylab = y(end-1);
        text(xlab, ylab, sprintf(' %.3g', m), ...
            'HorizontalAlignment','left','VerticalAlignment','middle');
    end
end
hold off

% Figure 2 (loglog): fit log10(regret) ~ m * log10(horizon) + b
figure(2); hold on
% Odd subset
x = horizons(odd_idx);  y = regret(odd_idx);
mask = (x > 0) & (y > 0);  x = x(mask);  y = y(mask);
if numel(x) >= 2
    X = log10(x);       Y = log10(y);
    p = polyfit(X, Y, 1);           m = p(1);
    Yhat = polyval(p, X);
    r2 = 1 - sum((Y - Yhat).^2) / sum((Y - mean(Y)).^2);
    if r2 > R2
        xlab = x(end-1);  % small multiplicative right shift on log axis
        ylab = y(end-1);
        text(xlab, ylab, sprintf(' %.3g', m), ...
            'HorizontalAlignment','left','VerticalAlignment','middle');
    end
end
% Even subset
x = horizons(even_idx);  y = regret(even_idx);
mask = (x > 0) & (y > 0);  x = x(mask);  y = y(mask);
if numel(x) >= 2
    X = log10(x);       Y = log10(y);
    p = polyfit(X, Y, 1);           m = p(1);
    Yhat = polyval(p, X);
    r2 = 1 - sum((Y - Yhat).^2) / sum((Y - mean(Y)).^2);
    if r2 > R2
        xlab = x(end-1);
        ylab = y(end-1);
        text(xlab, ylab, sprintf(' %.3g', m), ...
            'HorizontalAlignment','left','VerticalAlignment','middle');
    end
end
hold off
