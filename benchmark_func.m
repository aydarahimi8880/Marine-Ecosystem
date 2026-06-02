function f = benchmark_func(x,name)

switch name

    case 'Sphere'
        f = sum(x.^2);

    case 'Rosenbrock'
        f = sum(100*(x(2:end)-x(1:end-1).^2).^2 + (x(1:end-1)-1).^2);

    case 'Rastrigin'
        f = sum(x.^2 - 10*cos(2*pi*x) + 10);

    case 'Ackley'
        n = length(x);
        f = -20*exp(-0.2*sqrt(sum(x.^2)/n)) ...
            - exp(sum(cos(2*pi*x))/n) ...
            + 20 + exp(1);



end

end
