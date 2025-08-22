clear; 
close all;

T_list = 31;   % 23:2:65
sigma_opt = [];
for T=T_list
    %% build full constraint jacobian for OPT 
    % T=31;
    J_full_opt = zeros(2*T, T);
    for t = 1:T
        i0 = 2*t-1;
        J_full_opt(i0+0, t) = 1;
        J_full_opt(i0+1, t) = -1;
        if t<T
            J_full_opt(i0+2, t) = -1; 
            J_full_opt(i0+3, t) = 1; 
        end
    end
    
    %% active jacobian and hessian
    
    % when T=31 odd number
    a_opt = [];
    for i = 2:T-2
        if mod(i,2) == 1
            a_opt=[a_opt,2*i-1];
        else
            a_opt=[a_opt,2*i];
        end
    end
    a_opt=[a_opt,2*T-1];
    J_active_opt = J_full_opt(a_opt,:);
    
    
    
    % hessian
    Q = 6 * eye(T);
        
    % Set the last diagonal entry to 4
    Q(T,T) = 4;
    
    % Set super- and sub-diagonal entries to -2
    for i = 1:T-1
        Q(i,i+1) = -2;  % super-diagonal
        Q(i+1,i) = -2;  % sub-diagonal
    end
    
    % right hand side
    S = zeros(T,1);
    
    % Odd indices: 
    S(1:2:end) = 20;
    
    % Even indices:
    S(2:2:end) = -20;
    
    
    %% full hessian
    H = [Q J_active_opt'; 
        J_active_opt, zeros(size(J_active_opt,1),size(J_active_opt,1))];
    R = [S; 4/5*ones(size(J_active_opt,1),1)];
    sigma_min = min(svd(H));
    
    % z_active = inv(H)*R;  % first T are primals, the rest T-2 are duals
    sigma_opt = [sigma_opt, sigma_min];


end
%% plot 

% singular value changes with respect to horizon length
figure;
plot(T_list, sigma_opt, 'o-', 'LineWidth', 1.5, 'MarkerSize', 6);

xlabel('Horizon length T');
ylabel('Smallest singular value');
title('Smallest singular value vs Horizon length');
grid on;

% matrix H
figure;

% Copy H, replace zeros with NaN so they won't be colored
H_masked = H;
H_masked(H_masked == 0) = NaN;

% Plot masked matrix
imagesc(H_masked, 'AlphaData', ~isnan(H_masked));

% Colormap and colorbar
colormap(jet);
colorbar;

xlabel('Column index');
ylabel('Row index');
title('Matrix H (zeros in white)');

grid on


return;

%% build full constraint jacobian for MPC
clear;

N=21;
J_full_mpc = zeros(2*N+2, N+1);
for t = 1:N+1
    i0 = 2*t-1;
    J_full_mpc(i0+0, t) = 1;
    J_full_mpc(i0+1, t) = -1;
    if t<N+1
        J_full_mpc(i0+2, t) = -1; 
        J_full_mpc(i0+3, t) = 1; 
    end
end

% active jacobian and hessian
a_mpc = [];
for i = 1:N
    if mod(i,2) == 1
        a_mpc=[a_mpc,2*i-1];
    else
        a_mpc=[a_mpc,2*i];
    end
end
J_active_mpc = J_full_mpc(a_mpc,:);


% hessian
Q_mpc = 6 * eye(N+1);  
% Set the last diagonal entry to 4
Q_mpc(N+1,N+1) = 4;
% Set super- and sub-diagonal entries to -2
for i = 1:N
    Q_mpc(i,i+1) = -2;  % super-diagonal
    Q_mpc(i+1,i) = -2;  % sub-diagonal
end


% right hand side
S_mpc = zeros(N+1,1);
% Odd indices: 
S_mpc(1:2:end) = 20;
% Even indices:
S_mpc(2:2:end) = -20;
S_mpc(end) = 0;

% full hessian
H = [Q_mpc J_active_mpc'; 
    J_active_mpc, zeros(size(J_active_mpc,1),size(J_active_mpc,1))];
R = [S_mpc; 4/5*ones(size(J_active_mpc,1),1)];
sigma_min = min(svd(H));

z_active = inv(H)*R;   % first N+1 are primals, the rest N are duals
