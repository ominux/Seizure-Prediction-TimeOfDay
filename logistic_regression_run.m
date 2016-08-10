function out = logistic_regression_run(W,data,bias)

if bias == 1
    data = [ones(size(data,1),1) data]';  % add the bias
end
out = sigmoid(W'*data);  % model output

end

function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = 1 ./ (1 + exp(-z));
end