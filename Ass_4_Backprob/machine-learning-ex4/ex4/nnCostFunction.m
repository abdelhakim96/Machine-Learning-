function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

% Setup some useful variables
m = size(X, 1);
         Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X=[ones(m,1) X];
yx=zeros(10,1);
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

Xt=X';
a1=zeros(size(Theta1,2),1);
a2=zeros(size(Theta2,2),1);
h=zeros(size(Theta2,1),1);
J=zeros(size(Theta2,1),1);
for i=1:m
    
a1=sigmoid(Theta1*Xt(:,i));
a1=[1;a1];
h=sigmoid(Theta2*a1);


yx=zeros(size(Theta2,1),1);
yx(y(i))=1;

%J=J+(1/m)*(-yx.*log(h))-(1-yx).*log(1-h);


J=J+(1/m)*((-yx.*log(h))-(1-yx).*log(1-h));

    
end


J=sum(J);
reg1=(lambda/(2*m))*sum(Theta1.^2,'All');
reg2=(lambda/(2*m))*sum(Theta2.^2,'All');
J=J+reg1+reg2;
size(J);

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

delta1=zeros(size(Theta1,1),size(Theta1,2));
delta2=zeros(size(Theta2,1),size(Theta2,2));
for i=1:m
a1=X(i,:);   
a1=a1'; 
a2=sigmoid(Theta1*a1);
a2=[1;a2];
a3=sigmoid(Theta2*a2);


yx=zeros(size(Theta2,1),1);

yx(y(i))=1;
d3=(a3-yx);


d2=Theta2'*d3.*((a2).*(1-a2));
d2=d2(2:end);


delta1=delta1+d2*a1';
delta2=delta2+d3*a2';


end
Theta1_grad=delta1/m;
Theta2_grad=delta2/m;


delta1;
delta2;
%Theta1_grad=delta1/m;
%Theta2_grad=delta2/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


grad_reg1=(lambda/m)*Theta1;
grad_reg2=(lambda/m)*Theta2;

Theta1_grad=Theta1_grad+grad_reg1;
Theta2_grad=Theta2_grad+grad_reg2;
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
