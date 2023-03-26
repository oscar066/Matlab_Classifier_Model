
% loading the preprocessed dataset
data = dlmread('preprocessed_research_rpt.csv', ',', 1, 0);

% checking if the dataset has any missing values
any(isnan(data(:)));

% splitting the data in X and y where y is our target variable
y = data(:, 10);

X = data(:, [1:9 11:end]);

% splitting the data into train and test set
% 70 train set and 30 test set
cv = cvpartition(size(X,1), 'HoldOut', 0.3);

idxTrain = training(cv);
idxTest = test(cv);

X_train = X(idxTrain, :);
y_train = y(idxTrain, :);

X_test = X(idxTest, :);
y_test = y(idxTest, :);

% creating a simple classification model 
mdl = fitcecoc(X_train, y_train, 'Learners', templateSVM('Standardize',true));

% Evaluating the model
y_pred = predict(mdl, X_test);

accuracy = mean(y_pred == y_test);
fprintf('Accuracy: %0.2f%%\n', accuracy * 100);
