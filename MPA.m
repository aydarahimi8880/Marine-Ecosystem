%%%%%%%%%%% MPA 
function [Top_predator_fit, Top_predator_pos, Convergence_curve, Prey] = MPA(SearchAgents_no, Max_iter, lb, ub, dim, fobj, initial_population)

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

   
    if rand() < FADs
        U = rand(SearchAgents_no, dim) < FADs;
        Prey = Prey + CF * ((Xmin + rand(SearchAgents_no, dim).*(Xmax - Xmin)) .* U);
    else
        r = rand(); Rs = size(Prey,1);
        stepsize = (FADs*(1-r) + r) * (Prey(randperm(Rs),:) - Prey(randperm(Rs),:));
        Prey = Prey + stepsize;
    end

    Iter = Iter + 1;
    Convergence_curve(Iter) = Top_predator_fit;
end

end
