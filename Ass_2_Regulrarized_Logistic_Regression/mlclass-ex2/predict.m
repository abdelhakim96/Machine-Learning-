function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);
z=zeros(m, 1);
g=zeros(m, 1);
l=size(X,2);
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%


for i=1:m
   for cc=1:l    
   z(i,1)=z(i,1)+theta(cc)*X(i,cc);
   end 
    
   g(i)=1/(1+exp((-z(i))));    
   
   if g(i)>0.5
       p(i) =1;
   end
   
   
end





% =========================================================================


end
