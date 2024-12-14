import pandas as pd
import numpy as np
import joblib
import os

# UTILITIES
def importdata(path):
    print('Importing data from ', os.path.basename(path))

    # Load data using pandas
    trainData = pd.read_excel(path, header=0)

    # Separate features and targets
    X = trainData.loc[:, ['inputNi', 'inputMn', 'inputCo', 'temp', 'pKa1', 'acidC', 'H2O2_conc', 'solidToLiquid','time']] #, 'HtoLi'
    y = trainData.loc[:,['xLi', 'xNi', 'xMn', 'xCo']]

    # Check data types and shapes
    # print('\n::: FEATURES :::')
    # print(X.head())
    print("Features (X):\t", X.shape)
    # print("X data type:\t", type(X))

    # print('\n::: TARGETS :::')
    # print(y.head())
    print("Targets  (y):\t", y.shape)
    # print("y data type:\t", type(y))

    return X, y

def genFeatures(X1,X2):
    # Code below closely follows the pseudocode in https://doi.org/10.1021/acs.jcim.1c00670
    print('Expanding features')    
    # Input arrays   
    X1 = X1.to_numpy()
    X2 = X2.to_numpy()
    
    print("Original X1:", X1.shape)
    print("Original X2:", X2.shape)

    # Calculate the number of rows in each input array
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    # Expand dimensions and repeat arrays
    X1 = X1[:, np.newaxis, :].repeat(n2, axis=1)
    X2 = X2[np.newaxis, :, :].repeat(n1, axis=0)


    # Visual representation of expanded arrays
    # print("Expanded X1:")
    # print(X1.shape)
    # print("\nExpanded X2:")
    # print(X2.shape)

    # Combine arrays
    X1X2_combined = np.concatenate([X1, X2, X1 - X2], axis=2)

    # Visual representation of combined array
    # print("\nCombined array:")
    # print(X1X2_combined.shape)

    # Reshape the combined array
    X1X2_combined = X1X2_combined.reshape(n1 * n2, -1)

    # Visual representation of reshaped array
    print("Combined + reshaped X:", X1X2_combined.shape)
    # print(X1X2_combined.shape)
    # print(type(X1X2_combined))
    # print(X1X2_combined[:3])
    return X1X2_combined

def twinPredictorHelper(X_train, X_test, y_train, y_test_minus_y_pred):
    n1 = X_test.shape[0]
    n2 = X_train.shape[0]
    y_pred_distribution = np.zeros((n1,n2))
    y_pred_mu  = np.zeros((X_test.shape[0], y_train.shape[1]))
    y_pred_std = np.zeros(y_pred_mu.shape)

    # iterate over the four columns of the testing data to compute predictions and stdev
    for i in range(y_test_minus_y_pred.shape[1]):
        currCol = y_test_minus_y_pred[:,i]
        currTrain = y_train.to_numpy()[:,i]

        # reshape the results to the appropriate format so that each row
        # corresponds to each element of the testing data, and each column
        # is the prediction anchored by the training data

        y_pred_distribution = currCol.reshape(n1,n2) + currTrain

        # Calculate the prediction as the average across anchored predictions
        y_pred_mu[:,i] = np.mean(y_pred_distribution, axis=1)

        # Calculate the deviation of the predictions
        y_pred_std[:,i] = np.std(y_pred_distribution, axis=1)

    return y_pred_mu, y_pred_std

# LOAD PRE-TRAINED MODELS

if __name__ == "__main__":
    # Load data that was used to train the model. This is needed for the pairwise models
    X_train = joblib.load('model/xtrain.gz')
    y_train = joblib.load('model/ytrain.gz')
    # X_test = joblib.load('model/xtest.gz')
    # y_test = joblib.load('model/ytest.gz')

    # Load points to predict
    X_test,_ = importdata('test_20240827.xlsx')

    # Load pre-trained PD-GBR model
    mdl = joblib.load('model/PGBR.gz')

# Make predictions

# expTest_X  = genFeatures(X_test, X_train)
# y_test_minus_y_pred = mdl[0].predict(expTest_X)

# y_pred_mu, y_pred_std = twinPredictorHelper(X_train, X_test, y_train, y_test_minus_y_pred)

# y_pred_mu = pd.DataFrame(y_pred_mu, columns=['xLi', 'xNi', 'xMn', 'xCo'])
# y_pred_std = pd.DataFrame(y_pred_std, columns=['xLi', 'xNi', 'xMn', 'xCo'])

# print('Saved predictions as pred_mu.txt')
# y_pred_mu.to_csv('pred_mu.txt', index=False,sep='\t', float_format='%.6f')
# y_pred_std.to_csv('pred_std.txt', index=False,sep='\t', float_format='%.6f')