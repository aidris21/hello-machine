% Digit Images categorized by neural net

% mydir = '/mnt/autofs/netHome/guest02/matlab/digit_images/';
% files = dir(fullfile(mydir, '*.jpeg'));
% digit_array = int32.empty(0,784);
% label_array = int32.empty(0,0);
% for i = 1:length(files)
%     l = length(files);
%     filename = files(i).name;
%     fullFileName = fullfile(mydir, filename);
%     fprintf(1, 'Now reading %s\n', fullFileName)
%     digit = imread(fullFileName);
%     digit = imresize(digit,[28,28]);
%     digit = im2bw(digit,0.5);
%     digit = imcomplement(digit);
%     digits = reshape(digit, [1,784]);
%     digit_array = [digit_array;digits];
%     imshow(digit)
% end
% 
% csvwrite('mnist_mytest.csv', digit_array)
% fprintf('done')



% Now that that's over with (btw, I put in the labels by hand, figured that
% was easier)
 
data = csvread('mnist_mytest.csv');
labels = data(:,1)';
office_x = data(:,2:end)';


Ypred = mnistnn(Xsub);
Ypred(:,1)
[~, Ypred] = max(Ypred);
sum(Ysub == Ypred) / length(Ysub)