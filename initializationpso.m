function pop = initializationpso(Npop, dim, lb, ub)
    pop = repmat(lb, Npop, 1) + rand(Npop, dim) .* (repmat(ub - lb, Npop, 1));
end
