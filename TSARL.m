function [Position, Score, Convergence, Positions] = TSARL(Search_Agents, Max_iterations, Lowerbound, Upperbound, dimensions, objective, initial_population)

Position = zeros(1, dimensions);
Score = inf;

% Exploration rate used in Q-learning action selection
epsilon = 0.1;
% Discount factor in Q-learning
gamma = 0.99;
% Learning rate for Q-learning
alpha = 0.1;


if nargin < 7 || isempty(initial_population)
    Positions = init(Search_Agents, dimensions, Upperbound, Lowerbound);
else
    Positions = initial_population;
end
% Initialize Q-tables for Q-learning
Qtable_explore_exploit = cell(Search_Agents,1);

% Initialize each Q-table (each is a 2x2 matrix corresponding to 2 states and 2 actions)
for idx = 1 : Search_Agents
    Qtable_explore_exploit{idx} = zeros(2,2);
   
end

Convergence = zeros(1, Max_iterations);
t=0;

while t<Max_iterations
    for i=1:size(Positions,1)
        
    
        Flag4Upperbound=Positions(i,:)>Upperbound;
        Flag4Lowerbound=Positions(i,:)<Lowerbound;
        Positions(i,:)=(Positions(i,:).*(~(Flag4Upperbound+Flag4Lowerbound)))+Upperbound.*Flag4Upperbound+Lowerbound.*Flag4Lowerbound;
    
        fitness=objective(Positions(i,:));
    
		if fitness<Score 
            Score=fitness; 
            Position=Positions(i,:);
        end
        
    end

  %Determine current states for Q-learning decisions
         state_explore_exploit = double(t > Max_iterations / 2);

           % Choose actions using Q-learning
        action_explore_exploit = choose_action(Qtable_explore_exploit{i}, ...
            state_explore_exploit,epsilon);
          old_fitness = objective(Positions(i,:));

            xmin=1;
            xmax=4;
            xr=xmin+rand()*(xmax-xmin);
            xr=fix(xr);

    NewPositions = Positions;
  
   for i=1:size(Positions,1)
        for j=1:size(Positions,2)
           state_explore_exploit = double(t > Max_iterations/2);

           
             A1=((rand()+rand())-(2*rand()))/xr; 
                
            c2=rand();
        if(i==1)
       if action_explore_exploit == 1
            d_pos=abs(Position(j)-c2*Positions(i,j));
            Positions(i,j)=Position(j)+A1*d_pos;
        else
            d_pos=abs(Position(j)-c2*Positions(i,j));
            Positions(i,j)=Position(j)-A1*d_pos;
        
        
        end
        else
            
            
        if action_explore_exploit ==1
            d_pos=abs(Position(j)-c2*Positions(i,j));
            Pos(i,j)=Position(j)+A1*d_pos;
        else 
        d_pos=abs(Position(j)-c2*Positions(i,j));
        Pos(i,j)=Position(j)-A1*d_pos;
        
       
        end
             Positions(i,j)=(Pos(i,j)+Positions(i-1,j))/(2 + rand());

        end
        
        end
       Positions = max(min(Positions, Upperbound), Lowerbound);
           new_fitness = objective(Positions(i,:));

           %Diversity
        diversity_score = norm(Positions - mean(Positions));

       %Reward
        reward = calculate_reward(old_fitness, new_fitness, diversity_score);

        %Update Q-table
        Qtable_explore_exploit{i} = update_Qtable(Qtable_explore_exploit{i}, state_explore_exploit, action_explore_exploit, reward, state_explore_exploit, alpha, gamma);

      
        if new_fitness < old_fitness
          NewPositions(i,:) = Positions(i,:);
            if new_fitness < Score
                Score = new_fitness;
                Positions = Positions;
            end
        end
            
        end
      Positions = NewPositions;
    t=t+1;
end

final_population = Positions;

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
