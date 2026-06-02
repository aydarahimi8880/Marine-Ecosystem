clc;
clear;
close all;

%% Load MAT file
S = load('mydata.mat');

% اگر متغیر داخل فایل table باشد
T = S.mydata;

% تبدیل table به ماتریس
data = table2array(T);

%% Split features and labels
Y = data(:,1);      % ستون کلاس
X = data(:,3:end);  % ویژگی‌ها

% تبدیل برچسب‌ها
Y(Y==2) = -1;
Y(Y==4) = 1;

num_runs = 5;

acc_train_all = zeros(num_runs,1);
acc_test_all  = zeros(num_runs,1);

for run = 1:num_runs

    %% Train/Test split
    cv = cvpartition(length(Y),'HoldOut',0.5);

    X_train = X(training(cv),:);
    Y_train = Y(training(cv));

    X_test = X(test(cv),:);
    Y_test = Y(test(cv));

    %% Separate classes
    X_pos = X_train(Y_train==1,:);
    X_neg = X_train(Y_train==-1,:);

    epsilon = 1e-6;
    opts = optimoptions('quadprog','Display','off');

    %% Hyperplane 1
    H1 = [X_pos ones(size(X_pos,1),1)];
    G1 = [X_neg ones(size(X_neg,1),1)];

    Q1 = H1'*H1 + epsilon*eye(size(H1,2));
    f1 = zeros(size(Q1,1),1);

    A1 = -G1;
    b1 = -ones(size(G1,1),1);

    u1 = quadprog(Q1,f1,A1,b1,[],[],[],[],[],opts);

    w1 = u1(1:end-1);
    b1 = u1(end);

    %% Hyperplane 2
    H2 = [X_neg ones(size(X_neg,1),1)];
    G2 = [X_pos ones(size(X_pos,1),1)];

    Q2 = H2'*H2 + epsilon*eye(size(H2,2));
    f2 = zeros(size(Q2,1),1);

    A2 = -G2;
    b2 = -ones(size(G2,1),1);

    u2 = quadprog(Q2,f2,A2,b2,[],[],[],[],[],opts);

    w2 = u2(1:end-1);
    b2 = u2(end);

    %% Prediction
    predict_fun = @(X)(abs(X*w1 + b1) - abs(X*w2 + b2)) < 0;

    Y_pred_train = 2*predict_fun(X_train) - 1;
    Y_pred_test  = 2*predict_fun(X_test)  - 1;

    %% Accuracy
    acc_train_all(run) = mean(Y_pred_train==Y_train)*100;
    acc_test_all(run)  = mean(Y_pred_test==Y_test)*100;

end

fprintf('\nMean Train Accuracy: %.2f%%\n',mean(acc_train_all));
fprintf('Mean Test Accuracy: %.2f%%\n',mean(acc_test_all));
