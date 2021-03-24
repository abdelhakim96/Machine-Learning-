function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
[a,b] = size(X);
% You need to return the following variables correctly 
J = 0;
h=zeros(m,1);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
h=zeros(m,1);
for i=1:m
for cc=1:b    
h(i,1)=h(i,1)+theta(cc)*X(i,cc);
end
end

for i=1:m
J=J+(1/(2*m))*(h(i)-y(i))^2;
end
% =========================================================================

end
