clc;
clear;
close all;

% Load and normalize data
load('data.mat');
data = table2array(data);
data_scaled = mapminmax(data', -1, 1)';
labels = data_scaled(:,1);
features = data_scaled(:,2:end);

% Split dataset
cv = cvpartition(size(data,1), 'HoldOut', 0.5);
train_data = data_scaled(~cv.test, :);
test_data  = data_scaled(cv.test, :);

X_train = train_data(:,2:end);
Y_train = train_data(:,1);
X_test  = test_data(:,2:end);
Y_test  = test_data(:,1);

% Split by class
x_pos = X_train(Y_train == 1, :);
x_neg = X_train(Y_train == -1, :);
m1 = size(x_pos,1);
m2 = size(x_neg,1);
D = [x_pos; x_neg];
e1 = ones(m1,1);
e2 = ones(m2,1);

% Parameters
c1 = 100;
c2 = 100;
gamma = 0.1;
epsilon = 1e-6;
Npop = 50; 
MaxIt = 500     ;
lb1 = zeros(1, m2);
ub1 = c1 * ones(1, m2);
lb2 = zeros(1, m1);
ub2 = c2 * ones(1, m1);

% Kernels
K_pos = rbf_kernel(x_pos, D, gamma);
K_neg = rbf_kernel(x_neg, D, gamma);
H = [K_pos, ones(m1, 1)];
G = [K_neg, ones(m2, 1)];
inv_HH = inv(H'*H + 0.9 * eye(size(H,2)));
inv_GG = inv(G'*G + 0.93 * eye(size(G,2)));
Q1 = G * inv_HH * G';
Q2 = H * inv_GG * H';
fobj1 = @(alpha1) 0.5 * alpha1(:)' * Q1 * alpha1(:) - e2' * alpha1(:);
fobj2 = @(alpha2) 0.5 * alpha2(:)' * Q2 * alpha2(:) - e1' * alpha2(:);

algo_names = { 'PSO','SSARL', 'WOARL','RLMPA'};
num_rounds = 1;
last_maxit_test_accs = [];  % Array to store test accuracy of last maxit for each round
last_maxit_test_precisions = []; % Array to store test precision of last maxit for each round
last_maxit_test_recalls = []; % Array to store test recall of last maxit for each round

for round = 1:num_rounds
    %fprintf('\n===== Round %d =====\n', round);
    
    for maxit = 1:MaxIt
      %  fprintf('\n  > MaxIt: %d\n', maxit);
        
        % Generate random population for maxit == 1 in each round
        if maxit == 1
            pop1 = rand(Npop, m2);  % New random population for alpha1
            pop2 = rand(Npop, m1);  % New random population for alpha2
        else
            % Use best algorithm from previous maxit in this round
            switch best_algo
               
                 case 'PSO'
                    [ ~, ~, pop1] = PSO(Npop, 1, lb1, ub1, m2, @(x) 0);
                    [ ~, ~, pop2] = PSO(Npop, 1, lb2, ub2, m1, @(x) 0);
                case 'SSARL'
                    [~, ~, ~, pop1] = SSA(Npop, 1, lb1, ub1, m2, @(x) 0);
                    [~, ~, ~, pop2] = SSA(Npop, 1, lb2, ub2, m1, @(x) 0);
                case 'WOARL'
                    [~, ~, ~, pop1] = WOA(Npop, 1, lb1, ub1, m2, @(x) 0);
                    [~, ~, ~, pop2] = WOA(Npop, 1, lb2, ub2, m1, @(x) 0);
                case 'RLMPA'
                    [~, ~, ~, pop1] = MPA(Npop, 1, lb1, ub1, m2, @(x) 0);
                    [~, ~, ~, pop2] = MPA(Npop, 1, lb2, ub2, m1, @(x) 0);
                
            end
        end
        
        best_acc = -inf;
        best_algo = '';
        
        for i = 1:length(algo_names)
            algo = algo_names{i};
           % fprintf('    > Running %s...\n', algo);
            
            % Execute algorithm with current population
            [alpha1, ~] = feval(algo, Npop, 1, lb1, ub1, m2, fobj1, pop1);
            [alpha2, ~] = feval(algo, Npop, 1, lb2, ub2, m1, fobj2, pop2);
            
            z1 = inv_HH * (G' * alpha1(:));
            z2 = -inv_GG * (H' * alpha2(:));
            w1 = z1(1:end-1);
            b1 = z1(end);
            w2 = z2(1:end-1); 
            b2 = z2(end);
            
            K_train = rbf_kernel(X_train, D, gamma);
            f1_train = K_train*w1 + b1;
            f2_train = K_train*w2 + b2;
            Y_train_pred = ones(size(Y_train));
            Y_train_pred(abs(f1_train) > abs(f2_train)) = -1;
            
            K_test = rbf_kernel(X_test, D, gamma);
            f1_test = K_test*w1 + b1;
            f2_test = K_test*w2 + b2;
            Y_test_pred = ones(size(Y_test));
            Y_test_pred(abs(f1_test) > abs(f2_test)) = -1;
            
            acc_train = mean(Y_train_pred == Y_train) * 100;
            acc_test = mean(Y_test_pred == Y_test) * 100;

            TP_train = sum(Y_train_pred == 1 & Y_train == 1);
            FP_train = sum(Y_train_pred == 1 & Y_train == -1);
            FN_train = sum(Y_train_pred == -1 & Y_train == 1);
            precision_train = TP_train / (TP_train + FP_train);
            recall_train    = TP_train / (TP_train + FN_train);
            
            % تست
            TP_test = sum(Y_test_pred == 1 & Y_test == 1);
            FP_test = sum(Y_test_pred == 1 & Y_test == -1);
            FN_test = sum(Y_test_pred == -1 & Y_test == 1);
            precision_test = TP_test / (TP_test + FP_test);
            recall_test    = TP_test / (TP_test + FN_test);
            


          %  fprintf('    TrainAcc: %.2f%%, TestAcc: %.2f%%\n', acc_train, acc_test);
            
            % Update best algorithm for this maxit
            if acc_test > best_acc
                best_acc = acc_test;
                best_algo = algo;
            end
        end
        
        % Store test accuracy for last maxit in this round
        if maxit == MaxIt
            last_maxit_test_accs = [last_maxit_test_accs; best_acc];
            last_maxit_test_precisions = [last_maxit_test_precisions; precision_test];
            last_maxit_test_recalls = [last_maxit_test_recalls; recall_test];
          %  fprintf('>> Test Accuracy for Last MaxIt in Round %d: %.2f%%\n', round, best_acc);
        end
    end
    
    % Print end of round message with round number
    fprintf('End of round %d: Final accuracy = %.2f%%\n', round, last_maxit_test_accs(end));
end

% Calculate and report final statistics
mean_last_maxit_acc = mean(last_maxit_test_accs);
min_last_maxit_acc = min(last_maxit_test_accs);
std_last_maxit_acc = std(last_maxit_test_accs);

mean_last_maxit_precision = mean(last_maxit_test_precisions);

mean_last_maxit_recall = mean(last_maxit_test_recalls);

fprintf('\n===== Summary =====\n');
fprintf('Mean of test accuracies for last MaxIt across rounds: %.2f%%\n', mean_last_maxit_acc);
%fprintf('Min of test accuracies for last MaxIt across rounds: %.2f%%\n', min_last_maxit_acc);
%fprintf('Std of test accuracies for last MaxIt across rounds: %.2f%%\n', std_last_maxit_acc);

fprintf('Mean Precision (last MaxIt): %.2f\n', mean_last_maxit_precision);

fprintf('Mean Recall (last MaxIt): %.2f\n', mean_last_maxit_recall);

function K = rbf_kernel(X, D, gamma)
K = zeros(size(X, 1), size(D, 1));
for i = 1:size(X, 1)
    for j = 1:size(D, 1)
        K(i,j) = exp(-gamma * norm(X(i,:) - D(j,:))^2);
    end
end
end