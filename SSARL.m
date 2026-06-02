function [FoodPosition,FoodFitness,Convergence_curve,SalpPositions]=SSARL(Npop,Max_iter,lb,ub,dim,fobj,initial_population)


% Exploration rate used in Q-learning action selection
epsilon = 0.1;
% Discount factor in Q-learning
gamma = 0.99;
% Learning rate for Q-learning
alpha = 0.1;


if numel(ub) == 1
    ub = ub * ones(dim,1);
end
if numel(lb) == 1
    lb = lb * ones(dim,1); 
end
if size(ub,2) == dim && size(ub,1) == 1
    ub = ub';
end
if size(lb,2) == dim && size(lb,1) == 1
    lb = lb'; 
end

Convergence_curve = zeros(1, Max_iter);
if nargin < 7 || isempty(initial_population)
    SalpPositions = initializationssa(Npop, dim, ub, lb);
else
    SalpPositions = initial_population;
end


FoodPosition=zeros(1,dim);
FoodFitness=inf;


%calculate the fitness of initial salps

for i=1:size(SalpPositions,1)
    SalpFitness(1,i)=fobj(SalpPositions(i,:));
end

[sorted_salps_fitness,sorted_indexes]=sort(SalpFitness);

for newindex=1:Npop
    Sorted_salps(newindex,:)=SalpPositions(sorted_indexes(newindex),:);
end

FoodPosition=Sorted_salps(1,:);
FoodFitness=sorted_salps_fitness(1);

% Initialize Q-tables for Q-learning
Qtable_explore_exploit = cell(Npop,1);

% Initialize each Q-table (each is a 2x2 matrix corresponding to 2 states and 2 actions)
for idx = 1 : Npop
    Qtable_explore_exploit{idx} = zeros(2,2);
   
end


%Main loop
l=2; % start from the second iteration since the first iteration was dedicated to calculating the fitness of salps
while l<Max_iter+1
    
    c1 = 2*exp(-(4*l/Max_iter)^2); % Eq. (3.2) in the paper
    
    for i=1:size(SalpPositions,1)
        SalpPositions= SalpPositions';

         %Determine current states for Q-learning decisions
         state_explore_exploit = double(l > Max_iter / 2);

           % Choose actions using Q-learning
        action_explore_exploit = choose_action(Qtable_explore_exploit{i}, ...
            state_explore_exploit,epsilon);
        
        if i<=Npop/2
            for j=1:1:dim
                c2=rand();
                
                %%%%%%%%%%%%% % Eq. (3.1) in the paper %%%%%%%%%%%%%%
                if action_explore_exploit == 1
                    SalpPositions(j,i)=FoodPosition(j)+c1*((ub(j)-lb(j))*c2+lb(j));
                else
                    SalpPositions(j,i)=FoodPosition(j)-c1*((ub(j)-lb(j))*c2+lb(j));
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
            
        elseif i>Npop/2 && i<Npop+1
            point1=SalpPositions(:,i-1);
            point2=SalpPositions(:,i);
            
            SalpPositions(:,i)=(point2+point1)/2; % % Eq. (3.4) in the paper
        end
        
        SalpPositions= SalpPositions';
    end
    
    for i=1:size(SalpPositions,1)
        
        Tp=SalpPositions(i,:)>ub';
        Tm=SalpPositions(i,:)<lb';
        SalpPositions(i,:)=(SalpPositions(i,:).*(~(Tp+Tm)))+ub'.*Tp+lb'.*Tm;
        
        SalpFitness(i)=fobj(SalpPositions(i,:));
        % Calculate a diversity score based on the distance from the population mean
        diversity_score = norm(SalpPositions(i,:) - mean(SalpPositions));
         % Compute the reward based on improvement and diversity
        reward = calculate_reward(SalpFitness(i), SalpFitness(i), diversity_score);

         % Update the Q-tables for both decision processes using the reward
        Qtable_explore_exploit{i} = update_Qtable(Qtable_explore_exploit{i}, ...
            state_explore_exploit, action_explore_exploit, reward,state_explore_exploit, alpha, gamma);
        
        if SalpFitness<FoodFitness
            FoodPosition=SalpPositions(i,:);
            FoodFitness=min(SalpFitness);
            
        end
    end
    
    Convergence_curve(l)=FoodFitness;
    l = l + 1;
end


% Function to choose an action based on the current Q-table
function action = choose_action(q_table, state, epsilon)
if rand() < epsilon
    action = round(rand());
else
    [~ , act] = max(q_table(state + 1  , :));
    action =  act - 1;
end
end

% Function to calculate the reward based on fitness improvement and diversity score
function reward = calculate_reward(old_fitness, new_fitness, diversity_score)
improvement = old_fitness - new_fitness;
if improvement > 0
    reward = improvement + 0.1 * diversity_score;
else
    reward = -0.5 * abs(improvement);
end
end

% Function to update the Q-table based on the Q-learning update rule
function q_table = update_Qtable(q_table, state, action, reward, next_state, alpha, gamma)

state_index = state + 1;
action_index = action + 1;
next_state_index = next_state + 1;

[~, best_next_action_index] = max(q_table(next_state_index, :));

q_table(state_index, action_index) = q_table(state_index, action_index) + ...
    alpha * (reward + gamma * q_table(next_state_index, best_next_action_index) - q_table(state_index, action_index));
end
end
