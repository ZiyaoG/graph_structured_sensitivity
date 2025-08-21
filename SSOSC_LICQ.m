clear; clc;

% Parameters
ucostflag = 0;
N = 7;
x0 = 0;
xi = @(t) (mod(t,2)==1)*4/5 + (mod(t,2)==0)*(-4/5);

% Decision variables
x = sdpvar(N,1);
constraints = [];
objective = 0;

% Build cost and constraints
for t = 1:N
    xtm1 = x0;  % x_{t-1}
    if t > 1
        xtm1 = x(t-1);
    end
    % Cost
    objective = objective + (x(t) - xi(t))^2;    % without cost on u
    if ucostflag == 1
        objective = objective + (x(t) - xi(t))^2 + (x(t) - xtm1)^2;
    end
    
    % Constraints:
    % x_t in [-1, 1]
    constraints = [constraints; x(t) <= 1; x(t) >= -1];
    
    % u_t = x_t - x_{t-1} ∈ {±4/5} ⇔ enforce via inequality:
    % u_t = x(t) - x(t-1), must be ±4/5 ⇒ encode as:
    u = x(t) - xtm1;
    constraints = [constraints;
        u <= 4/5;
        -u <= 4/5];
    % constraints = [constraints;
    %     -u <= 4/5];
end

% Solve the optimization problem
options = sdpsettings('solver','gurobi','verbose',0);
sol = optimize(constraints, objective, options);

assert(sol.problem == 0, 'Optimization failed!');

% Primal solution
xopt = value(x)

% Dual solution (Lagrange multipliers of inequality constraints)
dual_vals = dual(constraints);

% Build full constraint Jacobian and identify active rows
n = N;
m = 4*N;
J = zeros(m, n);
row = 1;

for t = 1:N
    % Constraints:
    % x_t - 1 ≤ 0      → ∇ = e_t
    J(row, t) = 1; row = row + 1;
    
    % -x_t - 1 ≤ 0     → ∇ = -e_t
    J(row, t) = -1; row = row + 1;
    
    % x_t - x_{t-1} - 4/5 ≤ 0
    J(row, t) = 1;
    if t > 1
        J(row, t-1) = -1;
    end
    row = row + 1;
    
    % -(x_t - x_{t-1} - 4/5) ≤ 0 ⇒ -x_t + x_{t-1} + 4/5 ≤ 0
    J(row, t) = -1;
    if t > 1
        J(row, t-1) = 1;
    end
    row = row + 1;
end

% Identify active constraints
tol = 1e-6;
active_flags = abs(dual_vals) > tol;

% Extract rows of the full constraint Jacobian corresponding to active constraints
J_active = J(active_flags,:);


% LICQ check = full row rank of active constraint Jacobian
rank_JA = rank(J_active);
fprintf('Rank of active constraint Jacobian: %d (should be %d <= %d)\n', rank_JA, size(J_active,1), n);

% LICQ is satisfied if full row rank
if rank_JA == size(J_active,1)
    disp('LICQ holds !');
else
    disp('LICQ fails !');
end

% Hessian of Lagrangian 
H = diag(2*ones(1,N));
% if adding the u terms in cost function
if ucostflag == 1
    H = diag(4*ones(1,N));
    H(N,N) = 2;
    H = H + diag(-2*ones(1,N-1),1) + diag(-2*ones(1,N-1),-1);
end

% Compute null space basis Z of J_active
Z = null(J_active);  % Columns of Z form orthonormal basis of null space

% Compute reduced Hessian
Hred = Z' * H * Z;

% SSOSC = reduced Hessian PD
eigs_Hred = eig(Hred);
fprintf('Smallest eigenvalue of reduced Hessian: %.4f\n', min(eigs_Hred));

if all(eigs_Hred > 0)
    disp('SSOSC holds !');
else
    disp('SSOSC fails !');
end


% === Additional check: Active & Degenerate constraints ===
% Reconstruct constraint values at solution
cvals = [];
for t = 1:N
    xt = xopt(t);
    xtm1 = x0;
    if t > 1
        xtm1 = xopt(t-1);
    end
    u = xt - xtm1;
    cvals = [cvals;
        xt - 1;
        -xt - 1;
        xt - xtm1 - 4/5;
        -(xt - xtm1 - 4/5)];
end

% Identify all active constraints (residual ~ 0)
active_all = abs(cvals) < 1e-6;
active_dual = abs(dual_vals) > 1e-6;
degenerate_flags = active_all & ~active_dual;
degenerate_indices = find(degenerate_flags);

fprintf('\nTotal active constraints: %d\n', sum(active_all));
fprintf('Degenerate active constraints (active but zero dual): %d\n', length(degenerate_indices));

% Print info
types = {'x_t ≤ 1', 'x_t ≥ -1', 'u_t ≤ 4/5', 'u_t ≥ -4/5'};
for idx = degenerate_indices'
    t = ceil(idx / 4);
    ctype = mod(idx-1, 4) + 1;
    fprintf('  Degenerate @ t = %d: %s\n', t, types{ctype});
end
