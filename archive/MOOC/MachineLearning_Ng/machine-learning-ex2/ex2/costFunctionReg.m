function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
grad = zeros(size(theta));
this = 0;
that = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% =============================================================

for i = 1:m
    h = sigmoid(X(i,:) * theta);
    this = this + y(i) * log(h) + (1 - y(i)) * log(1 - h);
    for j = 1:(size(theta))
        grad(j) = grad(j) + (h - y(i)) * X(i,j);
    end
end

temp = (1 / m) * grad(1);
grad = (1 / m) * grad + (lambda / m) * theta;
grad(1) = temp;

for k = 1:size(theta)
    that = that + grad(k)^2;
end

J = - (1 / m) * this - (1 / (2 * m)) * that * lambda;

end
