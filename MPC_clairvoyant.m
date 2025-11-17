clear; close all;

%% -------------------- SETTINGS --------------------
% Problem sizes
T      = 501;                      % clairvoyant horizon (keep modest to run fast)
Np_list = [1:10 11:2:19];
x0_grid = -0.5:0.05:0.5;            % initial-state mesh

% Weights and bounds (same stage structure as your code)
xlim   = 1.0;                      % state bound
ulim   = 4/5;                      % move bound
xi_val = 4/5;                      % reference amplitude
u_flag = 0;                        % 1 adds move penalty, 0 constrain only
Pterm  = 1;                        % terminal cost for clairvoyant

% Build reference long enough for largest horizon
Hmax = max([T, max(Np_list)]);
xi_full = repmat(xi_val, Hmax, 1);
xi_full(2:2:end) = -xi_val;

% YALMIP + Gurobi options
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

%% -------------------- BUILD OPTIMIZERS --------------------
% MPC optimizers for each horizon in Np_list: no terminal cost/constraint
FMPC = containers.Map('KeyType','double','ValueType','any');
for h = Np_list(:)'
    x_seq = sdpvar(h+1,1);     % x(1..h+1)
    p0    = sdpvar(1,1);       % current state
    p_xi  = sdpvar(h,1);       % ref over next h stages

    cons = [];
    obj  = 0;

    % stage tracking and state bounds, t=1..h
    for t = 1:h
        obj  = obj + (x_seq(t) - p_xi(t))^2;
        cons = [cons, -xlim <= x_seq(t) <= xlim];
    end
    cons = [cons, -xlim <= x_seq(h+1) <= xlim];  % bound terminal state too

    % move limits and optional penalties, t=1..h+1
    for t = 1:h+1
        if t==1
            cons = [cons, -ulim <= x_seq(1)-p0 <= ulim];
            if u_flag, obj = obj + (x_seq(1)-p0)^2; end
        else
            cons = [cons, -ulim <= x_seq(t)-x_seq(t-1) <= ulim];
            if u_flag, obj = obj + (x_seq(t)-x_seq(t-1))^2; end
        end
    end

    FMPC(h) = optimizer(cons, obj, options, [p0; p_xi], x_seq);
end

% Clairvoyant optimizer with horizon T and terminal cost P=1
x_seq = sdpvar(T,1);
p0    = sdpvar(1,1);
p_xiT = sdpvar(T,1);

cons = [];
obj  = 0;
for t = 1:T
    obj  = obj + (x_seq(t) - p_xiT(t))^2;
    cons = [cons, -xlim <= x_seq(t) <= xlim];
end
for t = 1:T
    if t==1
        cons = [cons, -ulim <= x_seq(1)-p0 <= ulim];
        if u_flag, obj = obj + (x_seq(1)-p0)^2; end
    else
        cons = [cons, -ulim <= x_seq(t)-x_seq(t-1) <= ulim];
        if u_flag, obj = obj + (x_seq(t)-x_seq(t-1))^2; end
    end
end
obj = obj + Pterm * (x_seq(T) - p_xiT(T))^2;
FCLV_T = optimizer(cons, obj, options, [p0; p_xiT], x_seq);

%% -------------------- GRID EVALUATION --------------------
nx  = numel(x0_grid);
nh  = numel(Np_list);
DiffU = NaN(nx, nh);      % |u_Np(x0) - u_T,P(x0)|

% Pre-extract fixed reference for clairvoyant horizon T
xi_T = xi_full(1:T);

for ix = 1:nx
    x0 = x0_grid(ix);

    % clairvoyant first input at this x0 (same for all Np)
    x_pred_clv = FCLV_T{[ x0; xi_T ]};
    u_clv = x_pred_clv(1) - x0;

    for j = 1:nh
        h = Np_list(j);
        Fh = FMPC(h);
        xi_h = xi_full(1:h);
        x_pred_mpc = Fh{[ x0; xi_h ]};
        u_mpc = x_pred_mpc(1) - x0;

        DiffU(ix,j) = abs(u_mpc - u_clv);
    end
end

%% ----- FIGURE 2: transparent surface, projections, best fit on yz (odd Np) -----
[X,Y] = meshgrid(x0_grid, Np_list);
Z = DiffU.';

% projections
mean_over_x0 = mean(DiffU, 1, 'omitnan');   % 1 x nh, yz-plane curve
mean_over_h  = mean(DiffU, 2, 'omitnan');   % nx x 1, xz-plane curve

figure(2); clf
surf(X, Y, Z, 'EdgeColor','none', 'FaceAlpha', 0.2); shading interp; hold on
colormap(parula); colorbar; grid on
scatter3(X(:), Y(:), Z(:), 36, 'ko', 'filled', 'MarkerFaceAlpha', 0.9);

xlabel('x_0'); ylabel('MPC horizon N_p'); zlabel('|u_{N_p} - u_{T,P=1}|');
title('First-input gap, transparent surface with projections and best fit (odd N_p)');

% walls for projections
x_wall = min(x0_grid);     % yz plane at x = x_wall
y_wall = min(Np_list);     % xz plane at y = y_wall

% yz-plane projection, average over x0, plotted at x = x_wall
plot3( x_wall*ones(1,numel(Np_list)), Np_list, mean_over_x0, 'r-', 'LineWidth', 2);

% xz-plane projection, average over horizon, plotted at y = y_wall
plot3( x0_grid, y_wall*ones(1,numel(x0_grid)), mean_over_h(:).', 'b-', 'LineWidth', 2);

% -------- best fit on yz plane using odd horizons only --------
odd_mask = mod(Np_list,2)==1;
h_odd    = Np_list(odd_mask);
z_odd    = mean_over_x0(odd_mask);

% helper for R^2
computeR2 = @(y,yh) 1 - sum((y-yh).^2) / max(eps, sum((y-mean(y)).^2));

% Model 1: z ≈ C / h  (through origin)
X1   = 1 ./ h_odd(:);
C1   = (X1' * X1) \ (X1' * z_odd(:));
z1   = C1 * X1;
R2_1 = computeR2(z_odd(:), z1);

% Model 2: z ≈ C / h^2  (through origin)
X2   = 1 ./ (h_odd(:).^2);
C2   = (X2' * X2) \ (X2' * z_odd(:));
z2   = C2 * X2;
R2_2 = computeR2(z_odd(:), z2);

% Model 3: z ≈ C * alpha^h  (log-linear on positive z)
pos   = z_odd > 0;
if nnz(pos) >= 2
    hp   = h_odd(pos);
    zp   = z_odd(pos);
    P    = polyfit(hp, log(zp), 1);       % log z = log C + h log alpha
    alpha= exp(P(1));
    C3   = exp(P(2));
    z3   = C3 * alpha.^hp;
    R2_3 = computeR2(zp, z3);
else
    R2_3 = -Inf; alpha = NaN; C3 = NaN;
end

% choose best model
[R2_best, id] = max([R2_1, R2_2, R2_3]);

h_fine = linspace(min(Np_list), max(Np_list), 400);
switch id
    case 1  % C/h
        z_fine = C1 * (1 ./ h_fine);
        fit_label = sprintf('best fit (odd): C/Np, R^2=%.3f', R2_best);
    case 2  % C/h^2
        z_fine = C2 * (1 ./ (h_fine.^2));
        fit_label = sprintf('best fit (odd): C/Np^2, R^2=%.3f', R2_best);
    case 3  % C * alpha^h
        z_fine = C3 * alpha.^h_fine;
        fit_label = sprintf('best fit (odd): C\\cdot\\alpha^{Np}, R^2=%.3f', R2_best);
end

% draw best-fit curve on yz wall
plot3( x_wall*ones(size(h_fine)), h_fine, z_fine, 'k--', 'LineWidth', 2);

legend({'gap surface','data points','avg over x_0 (yz)','avg over N_p (xz)', fit_label}, ...
       'Location','northeastoutside');

view(45, 30);



%% -------------------- Console summary --------------------
fprintf('Grid size: %d x0 values × %d horizons = %d pairs\n', nx, nh, nx*nh);
fprintf('Mean |Δu| by Np:\n');
disp(array2table([Np_list(:), mean_over_x0(:)], ...
    'VariableNames', {'Np','MeanAbsGap'}));
