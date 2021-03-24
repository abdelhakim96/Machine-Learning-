function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
h=theta(1)*X(:,1)+theta(2)*X(:,2);
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


[a,b] = size(X);
%h=zeros(m,1);
grad=zeros(b,1);  
h=zeros(m,1);


for iter = 1:num_iters    % number of iterations 
grad(:)=0;  
h(:)=0;

%calculate h(theta)
for i=1:m
for cc=1:b    
h(i,1)=h(i,1)+theta(cc)*X(i,cc);
end
end


% calculate gradient f(h)
for k=1:b
for i=1:m
       
     grad(k)=grad(k)+alpha*(1/m)*(h(i)-y(i))*X(i,k);
    
end
end
 

 theta=theta-grad;

 
 
 
 J_history(iter) = computeCostMulti(X, y, theta);
  
    

end
