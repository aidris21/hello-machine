% Toy neural network

x = [0,0,1;
    1,1,1;
    1,0,1;
    0,1,1]; % Inputs

y = [0;1;1;0]; % Outputs
syn0 = randn(3,1);

for i = 1:60000
    l0 = x; % Inputs
    l1 = 1./(1+exp(-(l0*syn0))); % Multiply inputs with weights and put through sigmoid
    l1_error = y - l1; % Cost Function
    l1_delta = l1_error.*(l1.*(1-l1)); % Where can we go to minimize?
    syn0 = syn0 + (l0'*l1_delta); % Update Weights
end
    
disp('Output After Training: ')
disp(l1)