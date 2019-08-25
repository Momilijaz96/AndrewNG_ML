function [  ] = Optional_Ex( X,y,Xval,yval,lambda )
%OPTIONAL_EX optional exercise for plotting the learning curves with
%randomly selected examples
%You can call this optional exercise form command line like this:
%Optional_Ex(X_poly,y,X_poly_val,yval,3);

mtrain = size(X,1);
mval =  size(Xval,1);
m = min([mtrain mval]);
err_val = zeros(m,1);
err_train =zeros(m,1);
repeat = 50;
for i = 1 : m
    i_train_err = 0;
    i_val_err = 0;

    for j=1:repeat
        %Train Error for i sampes averaged for repeat times
        rand_indx = randi(m,1,i);
        Xrand = X(rand_indx,:);
        yrand = y(rand_indx);
        theta = trainLinearReg(Xrand,yrand,lambda);
        htrain = Xrand*theta;
        i_train_err = i_train_err + sum((htrain-yrand).^2)/(2*i);
        %Validation error for i samples averaged averaged for repeat times
        rand_indx = randi(m,1,i);
        Xvalrand = Xval(rand_indx,:);
        yvalrand = yval(rand_indx);
        hval = Xvalrand*theta;
        i_val_err= i_val_err + sum((hval - yvalrand).^2)/(2*i);
    end
    
    err_val(i)=i_val_err/repeat;
    err_train(i)=i_train_err/repeat;
end

plot(1:m, err_train, 1:m, err_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

