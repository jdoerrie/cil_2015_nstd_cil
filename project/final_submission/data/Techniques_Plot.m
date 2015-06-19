% Plots different techniques discussed in the report (mean and std).
% Data files have to be loaded/imported first before using this script.

fig = figure('Position', [200, 200, 600, 400]);

a = errorbar(log2(GetDataSVD(:, 1)), GetDataSVD(:, 2), GetDataSVD(:, 3), GetDataSVD(:, 3)); 
hold on;
b = errorbar(log2(GetDataFactNgbrItem(:, 1)), GetDataFactNgbrItem(:, 2), GetDataFactNgbrItem(:, 3), GetDataFactNgbrItem(:, 3)); 
hold on;
c = errorbar(log2(GetDataFactNgbrUser(:, 1)), GetDataFactNgbrUser(:, 2), GetDataFactNgbrUser(:, 3), GetDataFactNgbrUser(:, 3)); 
hold on;
d = errorbar(log2(GetDataregSVD(:, 1)), GetDataregSVD(:, 2), GetDataregSVD(:, 3), GetDataregSVD(:, 3));
hold on;
e = errorbar(log2(GetDataSVDpp(:, 1)), GetDataSVDpp(:, 2), GetDataSVDpp(:, 3), GetDataSVDpp(:, 3)); 
hold on;

leg = legend([a b c d e], 'Baseline SVD','Item-Item','User-User', 'RegSVD', 'SVD++'); 
set(leg,'FontSize',16); 
xlabel('log_{2}(K)', 'FontSize', 16); 
ylabel('RMSE', 'FontSize', 16); 