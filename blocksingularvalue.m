% Initialize range of N
N_values = [3:20,25:5:100,100:100:1000,5e3];
min_singular_values = zeros(size(N_values));

for idx = 1:length(N_values)
    N = N_values(idx);
    
    % Initialize matrix A
    A = zeros(N, N);
    A(1, 1) = 1;  % First row is all ones

    % Fill subsequent rows with the shifting pattern
    for i = 2:N
        if i <= N
            A(i, i-1) = -1;  % subdiagonal
            A(i, i) = 1;     % diagonal
        end
    end

    % Compute smallest singular value
    s = svd(A);
    min_singular_values(idx) = s(end);
end
%%
% Plot 1: Regular axis
figure;
plot(N_values, min_singular_values, '-o', 'LineWidth', 2);
xlabel('N');
ylabel('Smallest Singular Value');
title('Smallest Singular Value vs N (Linear Scale)');
grid on;

% Plot 2: Y-axis log scale
figure;
semilogy(N_values, min_singular_values, '-o', 'LineWidth', 2);
xlabel('N');
ylabel('Smallest Singular Value (log scale)');
title('Smallest Singular Value vs N (Y Log Scale)');
grid on;

% Plot 3: Both X and Y axis log scale
figure;
loglog(N_values, min_singular_values, '-o', 'LineWidth', 2);
xlabel('N (log scale)');
ylabel('Smallest Singular Value (log scale)');
title('Smallest Singular Value vs N (Log-Log Scale)');
grid on;


%% Compute slope in log-log space
log_N = log(N_values)';
log_sigma = log(min_singular_values)';

% Linear regression: log_sigma = slope * log_N + intercept
p = polyfit(log_N, log_sigma, 1);
slope = p(1);
intercept = p(2);

% Display slope
fprintf('Estimated slope in log-log scale: %.4f\n', slope);

% Hold on to the existing log-log plot
hold on;

% Generate fitted line values
N_fit = linspace(min(N_values), max(N_values), 100);
log_N_fit = log(N_fit);
log_sigma_fit = polyval(p, log_N_fit);
sigma_fit = exp(log_sigma_fit);

% Plot fitted line
loglog(N_fit, sigma_fit, '--r', 'LineWidth', 2);

% Add legend
legend('Data', sprintf('Fit: slope = %.2f', slope), 'Location', 'southwest');

