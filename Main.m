Npop = 20;
MaxIt = 50;
dim = 30;

[mean_best,std_best, best_all] = Ecosystem_Benchmark('Ackley', Npop, MaxIt, dim);

disp(best_all)
disp(std_best)
