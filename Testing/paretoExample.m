clear;
clc;
close all;

% Generating sample data for the demonstration
rng(10101); % For reproducibility
total_points = 100;
objectives_1 = rand(1, total_points) * 100;
objectives_2 = rand(1, total_points) * 100;

% Combining the objectives
objectives = [objectives_1; objectives_2]';

% Identifying Pareto front points
pareto = identify_pareto(objectives);
pareto_front = objectives(pareto, :);
non_pareto_front = objectives(~pareto, :);

% Plotting the Pareto front
figure;
scatter(non_pareto_front(:, 1), non_pareto_front(:, 2), 'blue', 'filled', 'DisplayName', 'Dominated Solutions');
hold on;
scatter(pareto_front(:, 1), pareto_front(:, 2), 'red', 'filled', 'DisplayName', 'Non-Dominated Solutions (Pareto Front)');
title('Pareto Front Demonstration');
xlabel('Objective 1');
ylabel('Objective 2');
legend('show');
grid on;

% Function to identify non-dominated points
function pareto_front = identify_pareto(scores)
    population_size = size(scores, 1);
    pareto_front = true(population_size, 1);
    for i = 1:population_size
        for j = 1:population_size
            if all(scores(j, :) <= scores(i, :)) && any(scores(j, :) < scores(i, :))
                pareto_front(i) = false;
                break;
            end
        end
    end
end