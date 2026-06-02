clc;
clear;
close all;

%% Load data from CSV
T = readtable("data.csv");

% Convert labels: M=1 , B=-1
Y = zeros(height(T),1);
Y(strcmp(T.diagnosis,'M')) = 1;
Y(strcmp(T.diagnosis,'B')) = -1;

% Features (remove id and diagnosis)
X = table2array(T(:,3:end));

num_runs = 30;

acc_train_all = zeros(num_runs,1);
acc_test_all = zeros(num_runs,1);

precision_train_all = zeros(num_runs,1);
precision_test_all = zeros(num_runs,1);

recall_train_all = zeros(num_runs,1);
recall_test_all = zeros(num_runs,1);

for run = 1:num_runs

%% Cross-validation (90% train, 10% test)
cv = cvpartition(length(Y),'HoldOut',0.5);

X_train = X(training(cv),:);
Y_train = Y(training(cv));

X_test = X(test(cv),:);
Y_test = Y(test(cv));

%% Separate classes
X_pos = X_train(Y_train == 1,:);
X_neg = X_train(Y_train == -1,:);

%% Parameters
epsilon = 1e-6;
opts = optimoptions('quadprog','Display','off');

%% Train hyperplane 1
H1 = [X_pos ones(size(X_pos,1),1)];
G1 = [X_neg ones(size(X_neg,1),1)];

Q1 = H1'*H1 + epsilon*eye(size(H1,2));
f1 = zeros(size(Q1,1),1);
A1 = -G1;
b1 = -ones(size(G1,1),1);

[u1,~,~,~] = quadprog(Q1,f1,A1,b1,[],[],[],[],[],opts);

w1 = u1(1:end-1);
b1 = u1(end);

%% Train hyperplane 2
H2 = [X_neg ones(size(X_neg,1),1)];
G2 = [X_pos ones(size(X_pos,1),1)];

Q2 = H2'*H2 + epsilon*eye(size(H2,2));
f2 = zeros(size(Q2,1),1);
A2 = -G2;
b2 = -ones(size(G2,1),1);

[u2,~,~,~] = quadprog(Q2,f2,A2,b2,[],[],[],[],[],opts);

w2 = u2(1:end-1);
b2 = u2(end);

%% Prediction
predict_fun = @(X) (abs(X*w1 + b1) - abs(X*w2 + b2)) < 0;

Y_pred_train = 2*predict_fun(X_train) - 1;
Y_pred_test = 2*predict_fun(X_test) - 1;

%% Accuracy
acc_train_all(run) = mean(Y_pred_train == Y_train)*100;
acc_test_all(run) = mean(Y_pred_test == Y_test)*100;

%% Precision & Recall (train)
TP = sum(Y_pred_train==1 & Y_train==1);
FP = sum(Y_pred_train==1 & Y_train==-1);
FN = sum(Y_pred_train==-1 & Y_train==1);

precision_train_all(run) = TP/(TP+FP);
recall_train_all(run) = TP/(TP+FN);

%% Precision & Recall (test)
TP = sum(Y_pred_test==1 & Y_test==1);
FP = sum(Y_pred_test==1 & Y_test==-1);
FN = sum(Y_pred_test==-1 & Y_test==1);

precision_test_all(run) = TP/(TP+FP);
recall_test_all(run) = TP/(TP+FN);

end

%% Average Results

fprintf('\n===== Average Results over %d runs =====\n',num_runs);

fprintf('Mean Test Accuracy: %.2f%%\n',mean(acc_test_all));
fprintf('Mean Test Precision: %.4f\n',mean(precision_test_all));
fprintf('Mean Test Recall: %.4f\n',mean(recall_test_all));
