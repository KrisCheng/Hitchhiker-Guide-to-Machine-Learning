function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
temp1 = 0;
temp2 = 0;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================
    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);
    if (iter > 1 && J_history(iter) > J_history(iter - 1))
        break;
    end
    for i = 1:m
        temp1 = temp1 - (theta(1) + X(i,2) * theta(2) - y(i));
        temp2 = temp2 - (theta(1) + X(i,2) * theta(2) - y(i)) * X(i,2);
    end
    theta(1) = theta(1) + alpha * temp1 / m;
    theta(2) = theta(2) + alpha * temp2 / m;
    temp1 = 0;
    temp2 = 0;
end

end
