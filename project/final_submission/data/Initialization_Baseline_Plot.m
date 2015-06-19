% Plots differently initialized baseline SVD results (mean and std).
% Data files have to be loaded first before using this script.

figure('Position', [100, 100, 600, 400]);

a = errorbar(Average_SVD(:, 2), Average_SVD(:, 3), 'color', [1, 0.5, 0.01]); 
hold on;
b = errorbar(AverageMean_SVD(:, 2), AverageMean_SVD(:, 3), 'color', [51/255 51/255 1]);
hold on;
c = errorbar(Compute_SVD(:, 2), Compute_SVD(:, 3), 'color', [0/255 204/255 204/255]); 
hold on;
d = errorbar(Learn_SVD(:, 2), Learn_SVD(:, 3), 'color', [102/255 102/255 0/255]); 
hold on;


leg = legend([a b c d], 'Average Bias SVD','Average Mean Bias SVD','Compute Bias SVD', 'Learn Bias SVD'); 
set(leg,'FontSize',16); 
xlabel('K', 'FontSize', 16); 
ylabel('RMSE', 'FontSize', 16); 