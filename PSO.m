%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPEA102
% Project Title: Implementation of Particle Swarm Optimization in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%
function [ BestFit, BestSol, final_population] = PSO(Npop, MaxIt, lb, ub, dim, fobj, initial_population)
%function [ BestFit, BestSol] = PSO(Npop, MaxIt, dim, fobj)
%[lb, ub] = get_bounds(func_name, dim);

w = 1;
wdamp = 0.98;
c1 = 1.5;
c2 = 2.0;

VelMax = 0.1 * (ub - lb);
VelMin = -VelMax;

% Init population
for i = 1:Npop
    if nargin < 7 || isempty(initial_population)
        particle(i).Position = lb + rand(1, dim) .* (ub - lb);
    else
        particle(i).Position = initial_population(i,:);
    end

    particle(i).Velocity = zeros(1, dim);
    particle(i).Cost = fobj(particle(i).Position);
    particle(i).Best.Position = particle(i).Position;
    particle(i).Best.Cost = particle(i).Cost;
end

% Global best
[~, gbest_idx] = min([particle.Cost]);
GlobalBest = particle(gbest_idx);

for it = 1:MaxIt
    for i = 1:Npop
        newVel = w * particle(i).Velocity ...
            + c1 * rand(1,dim) .* (particle(i).Best.Position - particle(i).Position) ...
            + c2 * rand(1,dim) .* (GlobalBest.Position - particle(i).Position);

        newVel = max(min(newVel, VelMax), VelMin);
        newPos = particle(i).Position + newVel;
        newPos = max(min(newPos, ub), lb);

        newCost = fobj(newPos);

      
        particle(i).Position = newPos;
        particle(i).Velocity = newVel;
        particle(i).Cost = newCost;

        if newCost < particle(i).Best.Cost
            particle(i).Best.Position = newPos;
            particle(i).Best.Cost = newCost;
            if newCost < GlobalBest.Cost
                GlobalBest = particle(i);
            end
        end
    end
    w = w * wdamp;
end

BestSol = GlobalBest.Position;
BestFit = GlobalBest.Cost;


final_population = zeros(Npop, dim);
for i = 1:Npop
    final_population(i,:) = particle(i).Position;
end

end
