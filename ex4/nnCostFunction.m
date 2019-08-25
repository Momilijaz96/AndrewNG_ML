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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%%%%%%%%%%%%%%%%%%%%%%SOLUTION BY MOMIL IJAZ%%%%%%%%%%%%%%%%%%%%%%%%%%
%Solution assumes there are three layers in NN
%% PART-1 COST FUNCTION -- J(THETA)

%Changing labels y to 10D vectors to transform 5000x1 to 5000x10
YVec=zeros(m,num_labels); %Keep it generic for any architecture of NN or dataset
for i=1:length(y)
    YVec(i,y(i))=1;
end

Msum=0;
for i=1:m
    %Feedforward propagation to compute h(x)
    %we need activations in col and theta in rows
    a1=X(i,:)';
    a1 =[1;a1]; %adding bias activation unit
    
    z2=Theta1*a1;  %z2=25x1
    a2=sigmoid(z2); %a2=25x1
    
    a2=[1;a2]; %a2=26x1

    z3=Theta2*a2; %z3=10x1
    h=sigmoid(z3); %h(x) a 10x1 or 10D vector output of Neural network for each sample
    
    %Computing Ksum
    Ksum=0;
    for k=1:num_labels
        temp = ( YVec(i,k)*(log(h(k))) ) + ( (1-YVec(i,k))*(log(1-h(k))) );
        Ksum=Ksum+temp;
    end
    
    %Computing Msum
    Msum = Msum + Ksum;
end

J= (-1/m) * (Msum);

%%%%%%% Regularization %%%%%%%%

Theta1reg=sum(sum(Theta1(:,2:end).^2)); %Reg term of theta1 skipping the bias units first column
Theta2reg=sum(sum(Theta2(:,2:end).^2));

reg_term = (lambda/(2*m))*(Theta1reg+Theta2reg);

J=J+reg_term;

%% PART-2 BACKPROPAGATION ALGORITHM 

for i=1:m
    %Feedforward propagation to compute h(x)
    %we need activations in col and theta in rows to calculate errors
    
    a1=X(i,:)';%a1=400x1
    a1 =[1;a1]; %adding bias activation unit
    
    z2=Theta1*a1;  %z2=25x1
    a2=sigmoid(z2); %a2=25x1
    
    a2=[1;a2]; %a2=26x1

    z3=Theta2*a2; %z3=10x1
    h=sigmoid(z3); %h(x) a 10x1 or 10D vector output of Neural network for each sample
    
    %Calculating delta/errors for each activation unit in each layer
    delta3 = h' - YVec(i,:); %delta3 = 1x10
    delta2 = (delta3 * Theta2(:,2:end)) .* (sigmoidGradient(z2')); %delta2 = 1x25
    
    %Calculating derivative of J(theta) w.r.t each param
    Theta1_grad = Theta1_grad + (delta2'*a1');
    Theta2_grad = Theta2_grad + (delta3'*a2');
end

Theta1_grad=Theta1_grad/m;
Theta2_grad=Theta2_grad/m;

%% PART-3 REGULARIZED GRADIENT = REGULARIZED NN

Theta1_grad_reg_term =[zeros(size(Theta1,1),1) (lambda/m).*(Theta1(:,2:end))]; %Skipping bias units' weights coloumn by adding zeros
Theta2_grad_reg_term= [zeros(size(Theta2,1),1) (lambda/m).*(Theta2(:,2:end))];

Theta1_grad = Theta1_grad + Theta1_grad_reg_term;
Theta2_grad = Theta2_grad + Theta2_grad_reg_term;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
