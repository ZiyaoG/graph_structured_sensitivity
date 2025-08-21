clear all
close all

%% 1) PARAMETERS
N_list        = 2:7;
s_list = [];
for N=N_list
    %% 2) BUILD FULL JACOBIAN J_full (4N rows × N cols)
    J_full = zeros(4*N, N);
    for t = 1:N
        i0 = 4*(t-1);
        % state-upper:   x_t - 1 <= 0  ⟹ grad =  e_t
        J_full(i0+1, t) =  1;
        % state-lower:  -x_t - 1 <= 0  ⟹ grad = -e_t
        J_full(i0+2, t) = -1;
        % control-upper: x_t - x_{t-1} -4/5 <=0
        if t==1
            % x(0)=0 ⇒ grad wrt x1 is +1
            J_full(i0+3,1) =  1;
            % control-lower: -(x1-0)-4/5 <=0 ⇒ grad wrt x1 is -1
            J_full(i0+4,1) = -1;
        else
            J_full(i0+3, t  ) =  1;  % +x_t
            J_full(i0+3, t-1) = -1;  % -x_{t-1}
            J_full(i0+4, t  ) = -1;  % -x_t
            J_full(i0+4, t-1) =  1;  % +x_{t-1}
        end
    end

    [subset, min_sv] = find_worst_Ha_physical(J_full, N);
    s_list=[s_list,min_sv];
end

%% plot

% Compute slope in log-log space
log_x = log(N_list(:));
log_y = log(s_list(:));
p2 = polyfit(log_x, log_y, 1);
slope2 = p2(1);

% Generate fitted line
x_fit = linspace(min(N_list), max(N_list), 100);
log_x_fit = log(x_fit);
log_y_fit = polyval(p2, log_x_fit);
y_fit = exp(log_y_fit);

% Plot log-log figure
figure;
loglog(N_list, s_list, 'bo', 'LineWidth', 2); hold on;
loglog(x_fit, y_fit, '--r', 'LineWidth', 2);
xlabel('N (log scale)');
ylabel('Smallest Singular Value (log scale)');
title('Log-Log Plot with Slope from N\_list vs s\_list');
legend('Data', sprintf('Fit: slope = %.2f', slope2), 'Location', 'southwest');
grid on;

% Print slope
fprintf('Estimated slope from N_list vs s_list: %.4f\n', slope2);


return

%% functions
function [best_subset, min_sigma] = find_worst_Ha_physical(H, k)
    % H: 4N x n constraint Jacobian
    % k: number of active rows to select
    m = size(H, 1);
    assert(mod(m, 4) == 0, 'Number of rows must be divisible by 4');
    N = m / 4;

    subsets = nchoosek(1:m, k);  % All possible row combinations
    min_sigma = inf;
    best_subset = [];

    for i = 1:size(subsets, 1)
        rows = subsets(i, :);
        if ~is_physically_feasible(rows, N)
            continue;
        end
        Ha = H(rows, :);
        s = svd(Ha);
        sigma_min = s(end);

        if sigma_min > 1e-12 && sigma_min < min_sigma
            min_sigma = sigma_min;
            best_subset = rows;
        end
    end
end

function ok = is_physically_feasible(rows, N)
    % Rows for 3rd and 4th constraint per time step
    third_constraints = 4 * (1:N) - 1;  % rows 3, 7, 11, ...
    fourth_constraints = 4 * (1:N);     % rows 4, 8, 12, ...

    % Logical arrays of length N: true if constraint is active at that time step
    active_third  = ismember(third_constraints, rows);
    active_fourth = ismember(fourth_constraints, rows);

    % Check for >2 consecutive actives for each
    bad_third  = conv(double(active_third),  [1 1 1], 'valid') == 3;
    bad_fourth = conv(double(active_fourth), [1 1 1], 'valid') == 3;

    ok = ~any(bad_third) && ~any(bad_fourth);
end
