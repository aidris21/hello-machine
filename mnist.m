%MNIST dataset analyzed in Matlab

train = csvread('mnist_train.csv', 1, 0);   % Load and read training data
sub = csvread('mnist_test.csv', 1, 0);     % Load and read submission data

% Previewing some of the images, mostly for fun

figure
colormap(gray)
for i = 1:20
    subplot(5,5,i)
    % IMPORTANT: Remember that apostraphe at the end, it will flip the
    % images the right way.
    digit = reshape(train(i, 2:end), [28,28])';                   
    imagesc(digit)
    title(num2str(train(i, 1)))
    set(gca, 'xtick', [])         % Get's rid of those pesky axes
    set(gca, 'ytick', [])         %
end


% Now let's get down to business. *cracks knuckles

labels = train(:, 1);
labels(labels == 0) = 10;  % Renaming the zeros to tens to simplify the loop
labels_dum = dummyvar(labels);  %The dummy variables for each label
pixels = train(:, 2:end);

%Transposing the data for input to Neural Network Toolbox

labels = labels';
labels_dum = labels_dum';
pixels = pixels';

% Sets the random number generator seed to 1. Many authors recommend this
% to be consistent for reproducibility. 
rng(1);

% Holds out a third of the data for subing. IMO, this data should be
% called subing, and the 'subing' data should be call verification data
n = size(train,1);
holdout = cvpartition(n, 'Holdout', int16(n/3));

% subing and training data
Xtrain = pixels(:, training(holdout));
Ytrain = labels_dum(:, training(holdout));
Xsub = pixels(:, test(holdout));
Ysub = labels(test(holdout));
Ysub_dum = labels_dum(test(holdout));






