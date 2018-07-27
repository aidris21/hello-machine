% Toy neural network

if exist('train') == 0 && exist('test')  == 0
    train = csvread('mnist_train.csv', 1, 0);   % Load and read training data 
    test = csvread('mnist_test.csv', 1, 0);     % Load and read  data
end
%===============================================================================================================================
syn0 = randn(784,16);
syn1 = randn(16,10);
b0 = randn(1,16);
b1 = randn(1,10);

epochs = 10;
alpha = 0.1;
%===============================================================================================================================

for i = 1:epochs
    batch = datasample(train,100,'Replace',false);
    x_batch = batch(:,2:end);
    x_batch = x_batch/255;
    y_batch = batch(:,1);
    y_batch(y_batch == 0) = 10;
    y_batch = y_batch';
    y_batch = full(ind2vec(y_batch));
    y_batch = y_batch';

    for i = 1:1000
        l0 = x_batch; % Inputs
        l1 = 1./(1+exp(-((l0*syn0)+b0))); % Multiply inputs with weights and put through sigmoid (layer 1)
        l2 = 1./(1+exp(-((l1*syn1)+b1))); % Multiply inputs with weights and put through sigmoid (layer 2)
        l3 = softmax(l2');
        l3 = l3';
        l3_error = mean(mean((y_batch - l3).^2)); % Cost Function
        l3_delta = 2*l3_error*(l3.*(1-l3)); % Where can we go to minimize?
        l1_error = l3_delta*(syn1'); %How much did each l1 value contribute to the l2 error (according to the weights)?
        l1_delta = l1_error.*(l1.*(1-l1)); % Where can we go to minimize?
        syn1 = syn1 + alpha*(l1'*l3_delta); % Update Weights
        syn0 = syn0 + alpha*(l0'*l1_delta); % Update Weights
        b0 = b0 - (mean(mean(l1_error)));
        b1 = b1 - l3_error;
        if mod(i,100) == 0
           % Can use to track progress of different parameters
        end
    end
    
%     output{i} = l2;

disp('Output After Training: ')
y_batch = vec2ind(y_batch')';
l3 = vec2ind(l3')';
sum(y_batch == l3) / length(y_batch)
l3_error

end
%==============================================================================================================================

% output_full = [];
% 
% for i = 1:epochs
%    
%     output_full = output_full + output{i};
%     
% end
% 
% output_full = output_full/epochs;
% 
% disp('Output After Training: ')
% y = vec2ind(y)';
% output_full = vec2ind(output_full)';
% sum(y == output_full) / length(y)

