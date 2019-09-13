function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

test_val = [0.01,0.03,0.1,0.3,1,3,10,30];
C_opt = C;
sigma_opt = sigma;

for C = test_val
    for sigma = test_val
        
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        pred = svmPredict(model,Xval);
        jval_err = mean(double(pred ~= yval)); %Classification error
        if (sigma == 0.01) && (C ==0.01)  %Initilizing min jval error as first error
            min_jval_err = jval_err;
        end    
        if min_jval_err >  jval_err %Checking error on each iteration for eah C sigma pair
            min_jval_err = jval_err;
            C_opt = C;
            sigma_opt = sigma;
        end
    end
end


C = C_opt; %Returning optimum C and sigma
sigma = sigma_opt;
    





% =========================================================================

end
