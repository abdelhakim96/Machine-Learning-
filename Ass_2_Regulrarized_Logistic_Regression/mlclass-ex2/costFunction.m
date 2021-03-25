function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
l=size(X,2);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
z=zeros(m);
h=zeros(m);
for i=1:m
for cc=1:l    
z(i,1)=z(i,1)+theta(cc)*X(i,cc);
end
end

for i=1:m
h(i)=1/(1+exp(-z(i)));

end
for i=1:m
J=J+(1/m)*(-y(i)*log(h(i))-(1-y(i))*log(1-h(i)));
end

for i=1:m
    for j=1:l
    grad(j)=grad(j)+(1/m)*(h(i)-y(i))*X(i,j);
    end
end




% =============================================================

end