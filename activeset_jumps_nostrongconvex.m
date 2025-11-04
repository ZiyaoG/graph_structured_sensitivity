% primals_jump_demo.m
% Show when primals jump at active-set changes for x+=x+u with box constraints.
% Case A: strictly convex (no jumps, only kinks).
% Case B: semidefinite Hessian (true jumps).

clear; clc;

T = 2;                 % keep it minimal, cost includes x(1)
xmax = 1.0; umax = 0.4;
x0   = 0.0;            % fixed initial condition
phi_grid = linspace(-0.20, 0.20, 201);  % parameter we sweep (terminal equality x_T = phi)

% Time-varying reference, choose something simple
xi = [0.2; -0.1];      % xi(1) is tracked at t=1

% ------- Choose weights for the two cases -------
% Case A: strictly convex
QA = [1; 1];           % Q(0), Q(1) (note: cost uses x(0) and x(1))
RA = [1];              % R(0)

% Case B: semidefinite so cost becomes linear in u0
%   - Remove curvature that involves u0: set Q(1)=0 and R(0)=0.
QB = [1; 0];
RB = [0];

% Storage
uA = zeros(numel(phi_grid),1); x1A = uA;
uB = uA; x1B = uA;

% Common YALMIP variables (re-used per solve)
x = sdpvar(T+1,1);   % x0, x1, x2
u = sdpvar(T,1);     % u0, u1 (but with T=2, only u0,u1; u1 unused by cost if we never include x2)
opts = sdpsettings('solver','gurobi','verbose',0, ...
    'gurobi.FeasibilityTol',1e-9,'gurobi.OptimalityTol',1e-9, ...
    'gurobi.BarConvTol',1e-12,'gurobi.NumericFocus',2,'gurobi.TuneTimeLimit',0);

% Static constraints
Fstatic = [];
Fstatic = [Fstatic, x(1) == x0];                 % fix x0
Fstatic = [Fstatic, x(2) == x(1) + u(1)];        % x1 = x0 + u0
% We do not use u(2) or x(3) with T=2; variables exist but are irrelevant
Fstatic = [Fstatic, -umax <= u(1) <= umax];
Fstatic = [Fstatic, -xmax <= x(2) <= xmax];

% Helper to build objective given Q,R and phi
build_obj = @(Q,R,phi) ( Q(1)*(x(1) - xi(1)).^2 + Q(2)*(x(2) - xi(2)).^2 + R(1)*u(1).^2 ) ...
                        + 0*x(3) + 0*u(2); % keep dimensions harmless

for k = 1:numel(phi_grid)
    phi = phi_grid(k);

    % Terminal equality x_T = phi (here T=2 -> x(3) but we structured cost to use only x(1))
    % To keep exactly the "same model" flavor, impose terminal equality on x(2) instead:
    % this still fits linear dynamics and keeps the example minimal.
    Feq = [x(2) == phi];    % we bind x1 to phi, so u0 sets phi directly (since x1 = x0 + u0)

    % ----- Case A -----
    objA = build_obj(QA,RA,phi);
    solA = optimize([Fstatic, Feq], objA, opts);
    assert(solA.problem==0, solA.info);
    uA(k)  = value(u(1));
    x1A(k) = value(x(2));

    % ----- Case B -----
    objB = build_obj(QB,RB,phi);           % linear in u(1), because Q(2)=0 and R(1)=0
    solB = optimize([Fstatic, Feq], objB, opts);
    assert(solB.problem==0, solB.info);
    uB(k)  = value(u(1));
    x1B(k) = value(x(2));
end

% Plots
figure; tiledlayout(2,1); 
nexttile; hold on; grid on;
plot(phi_grid, uA, 'LineWidth', 1.5);
plot(phi_grid, uB, 'LineWidth', 1.5);
ylabel('u_0^*(\phi)');
legend('Case A: strict convex','Case B: semidefinite','Location','best');
title('Control vs parameter');

nexttile; hold on; grid on;
plot(phi_grid, x1A, 'LineWidth', 1.5);
plot(phi_grid, x1B, 'LineWidth', 1.5);
xlabel('\phi'); ylabel('x_1^*(\phi)');
legend('Case A: strict convex','Case B: semidefinite','Location','best');
title('State vs parameter');

% Print quick diagnostics around the switch
[~, idx0] = min(abs(phi_grid - 0));   % near phi=0
fprintf('Near phi=0:  uA=%+.3f, uB=%+.3f  |  x1A=%+.3f, x1B=%+.3f\n', uA(idx0), uB(idx0), x1A(idx0), x1B(idx0));
