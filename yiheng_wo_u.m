% Temporal MPC with YALMIP Optimizer and Gurobi

% PARAMETERS
T_sim = 100;            % Simulation horizon
Np = 10;                % Prediction horizon
x0 = 0;                 % Initial condition

% Build full reference signal xi_full for simulation
xi_full = repmat(4/5, T_sim+Np, 1);
xi_full(2:2:end) = -4/5;

% Define YALMIP variables
x_seq = sdpvar(Np, 1);      % Predicted state sequence
p0    = sdpvar(1, 1);       % Current state (parameter)
p_xi  = sdpvar(Np, 1);    % Reference over prediction horizon

% Build constraints and objective symbolically
cons = [];
obj  = 0;
for t = 1:Np
    % Quadratic tracking cost
    obj = obj + (x_seq(t) - p_xi(t))^2;
    % State bounds
    cons = [cons, -1 <= x_seq(t) <= 1];
    % Difference constraints
    if t == 1
        cons = [cons, x_seq(1) - p0 <= 4/5, p0 - x_seq(1) <= 4/5];
    else
        cons = [cons, x_seq(t) - x_seq(t-1) <= 4/5, x_seq(t-1) - x_seq(t) <= 4/5];
    end
end

% Compile optimizer to speed up repeated solves
options = sdpsettings('solver', 'gurobi', 'verbose', 0);
F = optimizer(cons, obj, options, [p0; p_xi], x_seq);

% SIMULATION LOOP (Receding Horizon)
x_sim = zeros(T_sim+1, 1);
x_sim(1) = x0;
for k = 1:T_sim
    % Prepare current state and future reference
    xi_pred = xi_full(k:k+Np-1);
    params = [x_sim(k); xi_pred];
    % Solve MPC QP
    x_pred = F{params};
    % Apply first step of predicted trajectory
    x_sim(k+1) = x_pred(1);
end

% PLOT RESULTS
figure;
plot(0:T_sim, x_sim, '-o', 'DisplayName', 'MPC state');
hold on;
plot(1:T_sim, xi_full(1:T_sim), '--', 'DisplayName', 'Reference \xi');
xlabel('Time step k');
ylabel('State x');
legend;
grid on;
