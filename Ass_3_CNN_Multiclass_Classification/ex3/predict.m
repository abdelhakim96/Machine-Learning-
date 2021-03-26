function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m=size(X,1);
X=[zeros(m,1) X];

m = size(X, 1);
num_labels = size(Theta2, 1);

ra1=size(Theta1,1);
ra2=size(Theta2,1);
a1=zeros(ra1,1);
a2=zeros(ra2,1);
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
M1=zeros(ra1, m);
M2=zeros(ra2, m);
size(Theta1)
size(X)
for i=1:m
%for j=1:ra1
%for k=1:size(X,2)    
    a1=sigmoid(Theta1*X(i,:)'); 
%end
%end

M1(:,i)=a1(:);

 size(Theta2)
 a1=[1; a1];   
 size(a1)
a2=sigmoid(Theta2*a1);
 
M2(:,i)=a2(:);



end

for i=1:m
    for j=1:ra2
        [number,pred]=max(M2(:,i));
        p(i)=pred;
        
    end
end










% =========================================================================


end
