function [Leader_pos, Leader_score, Convergence_curve, Positions] = WOARL(SearchAgents_no, Max_iter, lb, ub, dim, fobj, initial_population)

% RL parameters
alpha = 0.1;
gamma = 0.99;
epsilon = 0.1;

% Leader init
Leader_pos = zeros(1,dim);
Leader_score = inf;
 
if nargin < 7 || isempty(initial_population)
    Positions = initializationwoa(SearchAgents_no, dim, ub, lb);
else
    Positions = initial_population;
end



Convergence_curve = zeros(1, Max_iter);

% Initialize Q-tables
QTables = cell(SearchAgents_no,1);
for i = 1:SearchAgents_no
    QTables{i} = zeros(2,2); % 2 states (stagnant/improving), 2 actions (explore/exploit)
end
LastImproved = ones(1, SearchAgents_no);

t = 0;

while t < Max_iter
    % Evaluate all agents
    for i = 1:SearchAgents_no
        % Bound check
        Positions(i,:) = max(min(Positions(i,:), ub), lb);

        fitness = fobj(Positions(i,:));

        if fitness < Leader_score
            Leader_score = fitness;
            Leader_pos = Positions(i,:);
        end
    end

    a = 2 - t * (2 / Max_iter);
    a2 = -1 + t * ((-1)/Max_iter);
    epsilon = max(0.01, epsilon * 0.99);  % decay

    % Update positions
    for i = 1:SearchAgents_no
        % Q-learning: state and action
        state = double((t - LastImproved(i)) > 5);
        action = choose_action(QTables{i}, state, epsilon);

        % Store old pos and fit for reward
        old_pos = Positions(i,:);
        old_fit = fobj(old_pos);

        % Movement logic
        new_pos = Positions(i,:);
        for j = 1:dim
            r1 = rand(); r2 = rand();
            A = 2*a*r1 - a;
            C = 2*r2;
            b = 1;
            l = (a2-1)*rand + 1;
            p = rand();

            if action == 0  % Exploration
                if p < 0.5
                    if abs(A) >= 1
                        rand_leader_index = randi(SearchAgents_no);
                        X_rand = Positions(rand_leader_index, :);
                        D_X_rand = abs(C * X_rand(j) - Positions(i,j));
                        new_pos(j) = X_rand(j) - A * D_X_rand;
                    else
                        D_Leader = abs(C * Leader_pos(j) - Positions(i,j));
                        new_pos(j) = Leader_pos(j) - A * D_Leader;
                    end
                else
                    distance2Leader = abs(Leader_pos(j) - Positions(i,j));
                    new_pos(j) = distance2Leader * exp(b * l) * cos(l * 2*pi) + Leader_pos(j);
                end
            else  % Exploitation
                D_Leader = abs(C * Leader_pos(j) - Positions(i,j));
                new_pos(j) = Leader_pos(j) - A * D_Leader;  
            end
        end

        % Bound check
        new_pos = max(min(new_pos, ub), lb);
        new_fit = fobj(new_pos);
        diversity = norm(new_pos - mean(Positions));

        % Reward & Q update
        reward = calculate_reward(old_fit, new_fit, diversity);
        next_state = double((t - LastImproved(i)) > 5);
        QTables{i} = update_Qtable(QTables{i}, state, action, reward, next_state, alpha, gamma);

        % Accept new pos if improved
        if new_fit < old_fit
            LastImproved(i) = t;
            Positions(i,:) = new_pos;
            if new_fit < Leader_score
                Leader_score = new_fit;
                Leader_pos = new_pos;
            end
        end
    end

    t = t + 1;
    Convergence_curve(t) = Leader_score;
end
end

%RL Functions 
function action = choose_action(q_table, state, epsilon)
    if rand() < epsilon
        action = randi([0,1]);
    else
        [~, a] = max(q_table(state+1, :));
        action = a - 1;
    end
end

function reward = calculate_reward(old_fit, new_fit, diversity)
    improvement = old_fit - new_fit;
    if improvement > 0
        reward = improvement + 0.2 * diversity;
    else
        reward = -0.5 * abs(improvement);
    end
end

function q_table = update_Qtable(q_table, state, action, reward, next_state, alpha, gamma)
    s = state + 1;
    a = action + 1;
    ns = next_state + 1;
    [~, best_next_action] = max(q_table(ns,:));
    q_table(s,a) = q_table(s,a) + ...
        alpha * (reward + gamma * q_table(ns, best_next_action) - q_table(s,a));
end