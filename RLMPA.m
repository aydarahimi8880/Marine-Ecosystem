%%%%%%%%%%% MPA 
function [Top_predator_pos, Top_predator_fit, Convergence_curve, Prey] = RLMPA(SearchAgents_no, Max_iter, lb, ub, dim, fobj, initial_population)

% Q-learning params
epsilon = 0.1;
gamma = 0.99;
alpha = 0.1;


Top_predator_pos = zeros(1, dim);
Top_predator_fit = inf;
Convergence_curve = zeros(1, Max_iter);
stepsize = zeros(SearchAgents_no, dim);
fitness = inf(SearchAgents_no, 1);

 
if nargin < 7 || isempty(initial_population)
    Prey = initializationmpa(SearchAgents_no, dim, ub, lb);
else
    Prey = initial_population;
end


%  RL Q-table for each agent (2 phase x 2 action) 
Qtable = cell(SearchAgents_no,1);
for idx = 1:SearchAgents_no
    % 2 state, 2 action (levy, brown)
    Qtable{idx} = zeros(2,2); 
end


Xmin = repmat(lb, SearchAgents_no, 1);
Xmax = repmat(ub, SearchAgents_no, 1);

Iter = 0;
FADs = 0.2;
P = 0.5;

while Iter < Max_iter

    for i = 1:SearchAgents_no
        Prey(i,:) = max(min(Prey(i,:), ub), lb);
        fitness(i) = fobj(Prey(i,:));
        if fitness(i) < Top_predator_fit
            Top_predator_fit = fitness(i);
            Top_predator_pos = Prey(i,:);
        end
    end

    %
    if Iter == 0
        fit_old = fitness; Prey_old = Prey;
    end

     state = double(Iter > Max_iter/2);
      % RL actions 
     action = choose_action(Qtable{i}, state, epsilon);

     pos = Prey(i,:);
        old_fitness = fitness(i);

    Inx = (fit_old < fitness);
    Indx = repmat(Inx,1,dim);
    Prey = Indx .* Prey_old + ~Indx .* Prey;
    fitness = Inx .* fit_old + ~Inx .* fitness;
    fit_old = fitness; Prey_old = Prey;

    % 
    Elite = repmat(Top_predator_pos, SearchAgents_no, 1);
    CF = (1 - Iter/Max_iter)^(2*Iter/Max_iter);
    RL = 0.05 * levy(SearchAgents_no, dim, 1.5);
    RB = randn(SearchAgents_no, dim);

    for i = 1:SearchAgents_no
        for j = 1:dim
            R = rand();
            if Iter < Max_iter/3
                % Phase 1 (exploration)
                stepsize(i,j) = RB(i,j)*(Elite(i,j) - RB(i,j)*Prey(i,j));
                Prey(i,j) = Prey(i,j) + P*R*stepsize(i,j);
            elseif Iter < 2*Max_iter/3
                % Phase 2 (mix)
                if i > SearchAgents_no/2
                    stepsize(i,j) = RB(i,j)*(RB(i,j)*Elite(i,j) - Prey(i,j));
                    Prey(i,j) = Elite(i,j) + P*CF*stepsize(i,j);
                else
                    stepsize(i,j) = RL(i,j)*(Elite(i,j) - RL(i,j)*Prey(i,j));
                    Prey(i,j) = Prey(i,j) + P*R*stepsize(i,j);
                end
            else
                % Phase 3 (exploitation)
                stepsize(i,j) = RL(i,j)*(RL(i,j)*Elite(i,j) - Prey(i,j));
                Prey(i,j) = Elite(i,j) + P*CF*stepsize(i,j);
            end
        end
    end

    % 
    if Iter == 0
        fit_old = fitness; Prey_old = Prey;
    end

    Inx = (fit_old < fitness);
    Indx = repmat(Inx,1,dim);
    Prey = Indx .* Prey_old + ~Indx .* Prey;
    fitness = Inx .* fit_old + ~Inx .* fitness;
    fit_old = fitness; Prey_old = Prey;

   
    if action == 1
        U = rand(SearchAgents_no, dim) < FADs;
        Prey = Prey + CF * ((Xmin + rand(SearchAgents_no, dim).*(Xmax - Xmin)) .* U);
    else
        r = rand(); Rs = size(Prey,1);
        stepsize = (FADs*(1-r) + r) * (Prey(randperm(Rs),:) - Prey(randperm(Rs),:));
        Prey = Prey + stepsize;
    end

   %Evaluate Pry
    new_fitness = fobj(Prey(i,:));

    % diversity
    diversity_score = norm(Prey(i,:) - mean(Prey, 1));

    % Reward
    reward = calculate_reward(old_fitness, new_fitness, diversity_score);
      Qtable{i} = update_Qtable(Qtable{i}, state, action, reward, state, alpha, gamma);
    Iter = Iter + 1;
    Convergence_curve(Iter) = Top_predator_fit;
end

end


% RL Functions
function action = choose_action(q_table, state, epsilon)
    if rand() < epsilon
        action = round(rand());
    else
        [~, act] = max(q_table(state+1,:));
        action = act - 1;
    end
end

function reward = calculate_reward(old_fitness, new_fitness, diversity_score)
    improvement = old_fitness - new_fitness;
    if improvement > 0
        reward = improvement + 0.1 * diversity_score;
    else
        reward = -0.5 * abs(improvement);
    end
end

%  Q-table update
function q_table = update_Qtable(q_table, state, action, reward, next_state, alpha, gamma)
    state_index = state + 1;
    action_index = action + 1;
    next_state_index = next_state + 1;
    [~, best_next_action_index] = max(q_table(next_state_index, :));
    q_table(state_index, action_index) = q_table(state_index, action_index) + ...
        alpha * (reward + gamma * q_table(next_state_index, best_next_action_index) - q_table(state_index, action_index));
end

