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
Theta1_unbiased = Theta1(:, 2:end);

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
Theta2_unbiased = Theta2(:, 2:end);
Theta_reg = [Theta1_unbiased(:);Theta2_unbiased(:)];

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J=0;
J_2 = 0;
acc_2 = 0;
acc_1 = 0;
for i = 1:m;
  y_bin = zeros(num_labels, 1);
  y_bin(y(i), 1) = 1;
  h1 = sigmoid([1 X(i,:)] * Theta1');
  h2 = sigmoid([1 h1] * Theta2');
  fn = h2';
  J_2 =J_2+sum(((-y_bin.*log(fn))-((1-y_bin).*log(1-fn))));
  %gradient calculations
  delta_3 = fn - y_bin;
  delta_2 = Theta2(:,2:end)'*delta_3.*(h1'.*(1-h1'));
  %accumulation of gradients
  acc_2 = acc_2 + (delta_3*[1 h1]);
  acc_1 = acc_1 + (delta_2*[1 X(i,:)]);
  delta_2 = delta_2*0;
  delta_3 = delta_3*0;
endfor
J = (J_2/m)+((lambda/(2*m))*sum(Theta_reg.^2));
Theta1_grad = zeros(size(Theta1));
Theta1_grad = acc_1./m + (lambda/m)*[zeros(size(acc_1,1),1), Theta1(:,2:end)];%[Theta1(:,1) acc_1./m];
Theta2_grad = zeros(size(Theta2));
Theta2_grad = acc_2./m + (lambda/m)*[zeros(size(acc_2,1),1), Theta2(:,2:end)];%[Theta2(:,1) acc_2./m];

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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
