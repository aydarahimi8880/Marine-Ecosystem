
clc;
clear;
close all;

% Load and preprocess data
load("mydata.mat");
data = table2array(mydata);

% Parameters
num_repeats = 5;
train_accuracy_all = zeros(num_repeats,1);
test_accuracy_all  = zeros(num_repeats,1);

train_precision_all = zeros(num_repeats,1);
train_recall_all    = zeros(num_repeats,1);

test_precision_all  = zeros(num_repeats,1);
test_recall_all     = zeros(num_repeats,1);

for repeat = 1:num_repeats
    fprintf('Repeat %d/%d\n', repeat, num_repeats);

    % Split into 70/30
    cv = cvpartition(size(data,1),'HoldOut',0.5);
    idx = cv.test;
    train_data = data(~idx,:);
    test_data  = data(idx,:);

    mtrain = size(train_data,1);
    mtest  = size(test_data,1);
    dataset = [train_data; test_data];

    % Normalize to [-1,1]
    [dataset_scale, ps] = mapminmax(dataset', -1,1);
    dataset_scale = dataset_scale';
    train_data = dataset_scale(1:mtrain,:);
    test_data  = dataset_scale(mtrain+1:end,:);

    % Features & labels
    X_train = train_data(:, 2:end);
    Y_train = train_data(:, 1);
    X_test  = test_data(:, 2:end);
    Y_test  = test_data(:, 1);

    % Split classes
    x_pos = X_train(Y_train==1, :);
    x_neg = X_train(Y_train==-1,:);
    m1 = size(x_pos,1);  
    m2 = size(x_neg,1);  
    D  = [x_pos; x_neg];

    % Hyper‑params
    c1 = 100;
    c2 = 100; 
    gamma = 0.1;
    epsilon = 1e-6;
    e1 = ones(m1,1);
    e2 = ones(m2,1);

    % Compute RBF kernels
     % RBF Kernels
    K_pos = zeros(m1, size(D, 1));
    for i = 1:m1
        for j = 1:size(D, 1)
            K_pos(i, j) = exp(-gamma * norm(x_pos(i,:) - D(j,:))^2);
        end
    end

    K_neg = zeros(m2, size(D, 1));
    for i = 1:m2
        for j = 1:size(D, 1)
            K_neg(i, j) = exp(-gamma * norm(x_neg(i,:) - D(j,:))^2);
        end
    end

    % Original Jayadeva form:
    H = [K_pos, ones(m1,1)];  
    G = [K_neg, ones(m2,1)];   

    % Precompute inverses
    inv_HH = inv(H' * H + epsilon*eye(size(H,2)));
    inv_GG = inv(G' * G + epsilon*eye(size(G,2)));

    % Q‑matrices in original dual form
    Q1 = G * inv_HH * G';   
    Q2 = H * inv_GG * H';   

    % Dual objectives (minimization):
    fobj1 = @(alpha1) 0.5*(alpha1(:)'*Q1*alpha1(:)) - (e2'*alpha1(:));
    fobj2 = @(alpha2) 0.5*(alpha2(:)'*Q2*alpha2(:)) - (e1'*alpha2(:));

    % SFOA parameters
    Npop = 50;
    Max_it = 500;
    nD1  = m2;   
    nD2  = m1;   

    % Optimize
    [alpha1_opt, ~] = WOARL(Npop, Max_it, 0, c1, nD1, fobj1);
    [alpha2_opt, ~] = WOARL(Npop, Max_it, 0, c2, nD2, fobj2);
    alpha1_opt = alpha1_opt(:);
    alpha2_opt = alpha2_opt(:);

    % Recover (w,b)
    z1 = inv_HH * (G' * alpha1_opt);       % hyperplane #1
    z2 = -inv_GG * (H' * alpha2_opt);      % hyperplane #2

    w1 = z1(1:end-1);
    b1 = z1(end);
    w2 = z2(1:end-1); 
    b2 = z2(end);

     % Kernel for test samples
    K_test = zeros(mtest, size(D, 1));
    for i = 1:mtest
        for j = 1:size(D, 1)
            K_test(i, j) = exp(-gamma * norm(X_test(i,:) - D(j,:))^2);
        end
    end

    %  kernel for train samples
    K_train = zeros(mtrain, size(D, 1));
    for i = 1:mtrain
        for j = 1:size(D, 1)
            K_train(i, j) = exp(-gamma * norm(X_train(i,:) - D(j,:))^2);
        end
    end


    f1_train = K_train*w1 + b1;
    f2_train = K_train*w2 + b2;
    Y_train_pred = ones(mtrain,1);
    Y_train_pred(abs(f1_train)>abs(f2_train)) = -1;

    f1_test = K_test*w1 + b1;
    f2_test = K_test*w2 + b2;
    Y_test_pred = ones(mtest,1);
    Y_test_pred(abs(f1_test)>abs(f2_test)) = -1;

    train_accuracy_all(repeat) = mean(Y_train_pred==Y_train)*100;
    test_accuracy_all(repeat)  = mean(Y_test_pred==Y_test)*100

    TP = sum(Y_train_pred==1 & Y_train==1);
    FP = sum(Y_train_pred==1 & Y_train==-1);
    FN = sum(Y_train_pred==-1 & Y_train==1);

    train_precision_all(repeat) = TP/(TP+FP);
    train_recall_all(repeat)    = TP/(TP+FN);

    % TEST
    TP = sum(Y_test_pred==1 & Y_test==1);
    FP = sum(Y_test_pred==1 & Y_test==-1);
    FN = sum(Y_test_pred==-1 & Y_test==1);

    test_precision_all(repeat) = TP/(TP+FP+eps);
    test_recall_all(repeat)    = TP/(TP+FN+eps);
end

% Summary
%fprintf('Mean Train Accuracy: %.2f%%\n', mean(train_accuracy_all));
fprintf('Mean Test  Accuracy: %.2f%%\n', mean(test_accuracy_all));
%fprintf('Min  Train Accuracy: %.2f%%\n', min(train_accuracy_all));
%fprintf('Min  Test  Accuracy: %.2f%%\n', min(test_accuracy_all));

fprintf('Mean Test  Precision: %.4f\n', mean(test_precision_all));
fprintf('Mean Test  Recall: %.4f\n', mean(test_recall_all));
