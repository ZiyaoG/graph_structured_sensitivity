%%========================================================================
%%  WORST‐CASE LICQ VIA TWO‐PHASE SVD + FEASIBILITY
%%========================================================================
clear all
close all

%% 1) PARAMETERS
N_list        = 2:8;
s_list = [];
for N=N_list
    %% 2) BUILD FULL JACOBIAN J_full (4N rows × N cols)
    % J_full = zeros(4*N, N);
    % for t = 1:N
    %     i0 = 4*(t-1);
    %     % state-upper:   x_t - 1 <= 0  ⟹ grad =  e_t
    %     J_full(i0+1, t) =  1;
    %     % state-lower:  -x_t - 1 <= 0  ⟹ grad = -e_t
    %     J_full(i0+2, t) = -1;
    %     % control-upper: x_t - x_{t-1} -4/5 <=0
    %     if t==1
    %         % x(0)=0 ⇒ grad wrt x1 is +1
    %         J_full(i0+3,1) =  1;
    %         % control-lower: -(x1-0)-4/5 <=0 ⇒ grad wrt x1 is -1
    %         J_full(i0+4,1) = -1;
    %     else
    %         J_full(i0+3, t  ) =  1;  % +x_t
    %         J_full(i0+3, t-1) = -1;  % -x_{t-1}
    %         J_full(i0+4, t  ) = -1;  % -x_t
    %         J_full(i0+4, t-1) =  1;  % +x_{t-1}
    %     end
    % end
    
    %% 3) BUILD FULL JACOBIAN J_full (half side constraint on u, 3N rows × N cols)
    J_full = zeros(3*N, N);
    for t = 1:N
        i0 = 3*(t-1);
        J_full(i0+1, t) = -1;
        J_full(i0+2, t) = 1;
        J_full(i0+3, t) = -1;
        if t>1
            J_full(i0+3, t-1) = 1;  % -x_{t-1}
        end
    end
    
    
    %% find the smallest singular value
    [subset, min_sv] = find_worst_Ha_physical(J_full, N);
    % fprintf('Worst-case subset: %s\n', mat2str(subset));
    % fprintf('Minimum nonzero singular value: %.4e\n', min_sv);
    s_list = [s_list,min_sv];

end
% block
H0=[1 0 0;
    -1 1 0;
    0 -1 1];
s0 = svd(H0);
sigma_min = s0(end);


return
%% functions
function [best_subset, min_sigma] = find_worst_Ha_physical(H, k)
    % H: 3N x n constraint Jacobian
    % k: number of active rows to select
    m = size(H, 1);
    assert(mod(m, 3) == 0, 'Number of rows must be divisible by 3');
    N = m / 3;

    subsets = nchoosek(1:m, k);
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
    % Check if constraint 3 is not active for >2 consecutive time steps
    third_constraints = 3 * (1:N);  % indices of 3rd constraint per timestep
    active_third = ismember(third_constraints, rows);  % logical array of length N

    % Check for more than 2 consecutive 1s in active_third
    bad = conv(double(active_third), [1 1 1], 'valid') == 3;
    ok = ~any(bad);
end
