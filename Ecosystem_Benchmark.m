function [mean_best,std_best, best_all] = Ecosystem_Benchmark(func_name,Npop,MaxIt,dim)

[lb, ub] = get_bounds(func_name, dim);
algo_names = {'PSO','SSA','MPA','WOA'};

Runs = 1;  
best_all = zeros(Runs,1);

for run = 1:Runs

    history = zeros(MaxIt,1);

    for it = 1:MaxIt
        
        if it == 1
            pop = rand(Npop,dim).*(ub-lb)+lb; % initial random population
        else
            switch best_algo
                case 'PSO'
                    [~, ~, pop] = PSO(Npop, 1, lb, ub, dim, @(x) benchmark_func(x,func_name), pop);
                case 'SSA'
                    [~, ~, ~, pop] = SSA(Npop, 1, lb, ub, dim, @(x) benchmark_func(x,func_name), pop);
                case 'MPA'
                    [~, ~, ~, pop] = MPA(Npop, 1, lb, ub, dim, @(x) benchmark_func(x,func_name), pop);
                case 'WOA'
                    [~, ~, ~, pop] = WOA(Npop, 1, lb, ub, dim, @(x) benchmark_func(x,func_name), pop);
            end
        end
        
        scores = zeros(length(algo_names),1);
        sols = cell(length(algo_names),1);
        pops = cell(length(algo_names),1);
        
        for i = 1:length(algo_names)
            algo = algo_names{i};
            switch algo
                case 'PSO'
                    [score, sol, newpop] = PSO(Npop, 10, lb, ub, dim, @(x) benchmark_func(x, func_name), pop);
                case 'SSA'
                    [score, sol, ~, newpop] = SSA(Npop, 10, lb, ub, dim, @(x) benchmark_func(x, func_name), pop);
                case 'MPA'
                    [score, sol, ~, newpop] = MPA(Npop, 10, lb, ub, dim, @(x) benchmark_func(x, func_name), pop);
                case 'WOA'
                    [score, sol, ~, newpop] = WOA(Npop, 10, lb, ub, dim, @(x) benchmark_func(x, func_name), pop);
            end
            scores(i) = min(score(:));
            sols{i} = sol;
            pops{i} = newpop;
        end
        
        [best_score, idx] = min(scores);
        best_algo = algo_names{idx};
        best_pos = sols{idx};
        pop = pops{idx};
        
        history(it) = best_score;
    end

    best_all = min(history);
end

mean_best = mean(best_all);
std_best = std(best_all); 

end
