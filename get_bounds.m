function [lb, ub] = get_bounds(func_name, dim)

switch func_name

    case 'Sphere'
        lb = -100 * ones(1,dim);
        ub =  100 * ones(1,dim);

    case 'Rosenbrock'
        lb = -100 * ones(1,dim);
        ub =  100 * ones(1,dim);

    case 'Rastrigin'
        lb = -5.12 * ones(1,dim);
        ub =  5.12 * ones(1,dim);

    case 'Ackley'
        lb = -32 * ones(1,dim);
        ub =  32 * ones(1,dim);


    otherwise
        error('Unknown benchmark function!');

end

end
