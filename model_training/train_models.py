import pandas as pd
import numpy as np
import joblib
import datetime
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
import time
import optuna
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import os  # Import the os module
import psutil
import sys

global kval
kval = 5
dataPath = "data/input_913pts_250407.xlsx"
models = []
statsTest = {'R2': [], 'MAE': [], 'MedAE': [], 'MSE': [], 'RMSE': []}
statsTrain = {'R2': [], 'MAE': [], 'MedAE': [], 'MSE': [], 'RMSE': [], 'Time': [], 'OptimTime': []}

# --- Unique Identifier Function ---
def generate_unique_id():
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    
    # Check for existing directories with the same date
    base_dir = 'out'
    # Ensure the base_dir exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_dir) if date_str in d and os.path.isdir(os.path.join(base_dir, d))]
    
    if not existing_dirs:
        return f"{date_str}_A"
    else:
        # Extract existing identifiers and increment
        identifiers = [d.split('_')[-1] for d in existing_dirs]
        last_identifier = max(identifiers)
        
        if last_identifier.isalpha():
            new_identifier = chr(ord(last_identifier) + 1)
        else:
            new_identifier = 'A'  # Reset if the identifier is not a letter
        
        return f"{date_str}_{new_identifier}"

# --- Create Unique Run Identifier ---
unique_run_id = generate_unique_id()
output_dir = os.path.join('out', unique_run_id)
os.makedirs(output_dir, exist_ok=True)

# Add a toggle for redirecting output
redirect_output = False

if redirect_output:
    # Redirect console output to log.txt
    class StreamToFile:
        def __init__(self, file_path):
            self.file = open(file_path, "w")
            self.last_message = None  # Track the last message written

        def write(self, message):
            # Filter out duplicate progress bar updates
            if message.strip() == self.last_message:
                return
            self.last_message = message.strip()

            # Write only the final progress bar update
            if "100%" in message or not message.strip().startswith("Best trial:"):
                self.file.write(message)
                self.file.flush()  # Ensure immediate writing to the file

        def flush(self):
            self.file.flush()

    log_file = os.path.join(output_dir, "log.txt")
    sys.stdout = StreamToFile(log_file)
    sys.stderr = sys.stdout

print(f"Output directory: {output_dir}")

# UTILITIES
def importdata(path, cols):

    # Load data using pandas
    trainData = pd.read_excel(path, header=0)

    # Separate features and targets
    X = trainData.loc[:, cols]
    y = trainData.loc[:,['xLi', 'xNi', 'xMn', 'xCo']]

    smiles = trainData.loc[:,['SMILES']]

    # Calculate properties for each data point
    properties = []
    for i, row in smiles.iterrows():
        props = get_properties_from_smiles(row['SMILES'])
        properties.append(props)
    properties_df = pd.DataFrame(properties)
    properties_df = properties_df.drop(columns=['smiles', 'error'], errors='ignore')

    # Concatenate properties with the original data
    X = pd.concat([X, properties_df], axis=1)

    # Check data types and shapes
    print('\n::: FEATURES ::')
    # print(X.head())
    # print("X columns:\t", X.columns)
    print("X shape:\t", X.shape)
    print("X data type:\t", type(X))

    print('\n::: TARGETS :::')
    print("y shape:\t", y.shape)
    print("y data type:\t", type(y))

    return X, y

def get_properties_from_smiles(smiles):
    """Calculates relevant properties for metal complex solubility and acidity."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        properties = {}
        # properties['smiles'] = smiles

        properties['num_atoms'] = mol.GetNumAtoms()
        properties['Mr_acid'] = Descriptors.MolWt(mol)

        properties['logP'] = Crippen.MolLogP(mol)
        properties['tpsa'] = Descriptors.TPSA(mol)
        properties['hbd'] = Lipinski.NumHDonors(mol)
        properties['hba'] = Lipinski.NumHAcceptors(mol)

        properties['max_partial_charge'] = Descriptors.MaxPartialCharge(mol)
        properties['min_partial_charge'] = Descriptors.MinPartialCharge(mol)

        # Calculate Polarizability and Molar Refractivity
        properties['molar_refractivity'] = Descriptors.MolMR(mol)

        propdict = Descriptors.CalcMolDescriptors(mol)

        # Carboxilic
        properties['fr_COO'] = propdict['fr_COO'] # Carboxylic acid
        properties['fr_C_O'] = propdict['fr_C_O'] # C-O bond

        # Amines
        properties['fr_NH0'] = propdict['fr_NH0'] # Primary amine
        properties['fr_NH2'] = propdict['fr_NH2'] # Tertiary amine

        # Halogens
        properties['fr_halogen'] = propdict['fr_halogen'] # Halogen atoms

        # Others
        properties['fr_Al_OH'] = propdict['fr_Al_OH'] # Aliphatic hydroxyl group

        # Count the number of sulfur atoms
        properties['fr_S'] = smiles.count('S')

        return properties
    else:
        return {'smiles': smiles, 'error': 'Invalid SMILES'}

def splitData(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=820)
    # Print data types and shapes
    print('\n::: SPLIT DATA :::')
    print('Training data:')
    print("X shape:\t", X_train.shape)
    print("y shape:\t", y_train.shape)
    print('Testing data:')
    print("X shape:\t", X_test.shape)
    print("y shape:\t", y_test.shape)
    
    return X_train, X_test, y_train, y_test

def saveModel(mdl, name: str, subfolder: str = None):
    if subfolder:
        model_dir = os.path.join(output_dir, subfolder)
    else:
        model_dir = output_dir
    os.makedirs(model_dir, exist_ok=True)
    filename = f"{name}.gz"
    filepath = os.path.join(model_dir, filename)
    joblib.dump(mdl, filepath)
    print(f'Model saved as {filepath}')

def saveXLSX(df: pd.DataFrame, name: str, subfolder: str = None):
    if subfolder:
        data_dir = os.path.join(output_dir, subfolder)
    else:
        data_dir = output_dir
    os.makedirs(data_dir, exist_ok=True)
    filename_xlsx = f"{name}.xlsx"
    filepath = os.path.join(data_dir, filename_xlsx)
    df.to_excel(filepath, index=False)
    print(f'Data saved as {filepath}')

def genFeatures(X1,X2):
# Code below closely follows the pseudocode in https://doi.org/10.1021/acs.jcim.1c00670
    print('* EXPANDED FEATURES *')    
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

    # Combine arrays
    X1X2_combined = np.concatenate([X1, X2, X1 - X2], axis=2)

    # Reshape the combined array
    X1X2_combined = X1X2_combined.reshape(n1 * n2, -1)

    print("Combined and reshaped array:", X1X2_combined.shape)
    return X1X2_combined

def genTargets(y1,y2):
    print('\n* EXPANDED TARGETS *') 
    # Adapted approach to correctly handle multiple outputs
    # Input arrays   
    y1 = y1.to_numpy()
    y2 = y2.to_numpy()

    print("Original y1:", y1.shape)
    print("Original y2:", y2.shape)
    
    if y1.shape != y2.shape:
        print('ERROR! VECTORS MUST BE SAME SHAPE')
        quit()

    temp = np.zeros((y1.shape[0]**2, y1.shape[1]))

    for i in range(y1.shape[1]):
        t1 = y1[:,i]
        t2 = y2[:,i]

        t_combined = (t1[:,np.newaxis]-t2[np.newaxis,:]).flatten()

        temp[:,i]=t_combined

    y1y2_combined = temp

    print("Combined array:", y1y2_combined.shape)
    return y1y2_combined

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

def computeStats(y_test, y_pred):
# Performance metrics
    r2 = r2_score(y_test, y_pred)
    RMSE = root_mean_squared_error(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    MedAE = median_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)

    return r2, RMSE, MAE, MedAE, MSE

# CONVENTIONAL MODELS
def RF_model(X_train, X_test, y_train, y_test, name: str):
    print('\n::: RANDOM FOREST :::')

    feature_names = X_train.columns.tolist()

    def objective(trial):
        # Define the hyperparameter search space
        n_estimators = trial.suggest_int('n_estimators', 10, 500)
        max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)

        # Define the model
        regr = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features='sqrt'
        )

        # Perform cross-validation
        kfold = KFold(n_splits=kval, shuffle=True)
        scores = cross_val_score(regr, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
        return scores.mean()

    print('Optimizing model hyperparameters...')
    print('> Goal: Maximize neg MSE')

    start_time = time.time()
    # Optimize hyperparameters using Optuna
    try:
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=100, show_progress_bar=True, n_jobs=-1)
    except Exception as e:
        print(f"Error during optimization: {e}")
    elapsed_time = (time.time() - start_time)
    statsTrain['OptimTime'].append(elapsed_time)

    print('\nOptimized parameters:')
    for key, value in study.best_params.items():
        print(f'  {key}\t{value}')

    best_model = RandomForestRegressor(
        n_estimators=study.best_params['n_estimators'],
        max_depth=study.best_params['max_depth'],
        min_samples_split=study.best_params['min_samples_split'],
        min_samples_leaf=study.best_params['min_samples_leaf'],
        max_features='sqrt'
    )
    best_model.feature_names_ = feature_names  # Store feature names

    start_time = time.time()
    print('\nFitting model with optimized hyperparameters...')
    best_model.fit(X_train, y_train)
    elapsed_time = (time.time() - start_time)
    print("Training time %.3f s" % elapsed_time)

    # Compute training stats
    models.append(name)
    y_pred = best_model.predict(X_train)

    r2, RMSE, MAE, MedAE, MSE = computeStats(y_train, y_pred)
    statsTrain['R2'].append(r2)
    statsTrain['RMSE'].append(RMSE)
    statsTrain['MAE'].append(MAE)
    statsTrain['MedAE'].append(MedAE)
    statsTrain['MSE'].append(MSE)
    statsTrain['Time'].append(elapsed_time)

    saveXLSX(pd.DataFrame(y_pred), name+'_train_ypred', subfolder='RF')

    # Compute test stats
    y_pred = best_model.predict(X_test)

    r2, RMSE, MAE, MedAE, MSE = computeStats(y_test, y_pred)
    statsTest['R2'].append(r2)
    statsTest['RMSE'].append(RMSE)
    statsTest['MAE'].append(MAE)
    statsTest['MedAE'].append(MedAE)
    statsTest['MSE'].append(MSE)

    # Save test predictions
    saveXLSX(pd.DataFrame(y_pred), name+'_test_ypred', subfolder='RF')
    
    return best_model

def GBR_model(X_train, X_test, y_train, y_test, name: str):
    print('\n::: HGBR :::')

    def objective(trial):
        # Define the hyperparameter search space
        max_iter = trial.suggest_int('max_iter', 50, 500)
        max_depth = trial.suggest_int('max_depth', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        max_bins = trial.suggest_int('max_bins', 64, 255)
        l2_regularization = trial.suggest_float('l2_regularization', 1E-6, 10, log=True)
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 20, 200)

        # Define the model
        regr = MultiOutputRegressor(
            estimator=HistGradientBoostingRegressor(
                max_iter=max_iter,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                learning_rate=learning_rate,
                max_bins=max_bins,
                l2_regularization=l2_regularization,
                max_leaf_nodes=max_leaf_nodes,
            )
        )

        # Perform cross-validation
        kfold = KFold(n_splits=kval, shuffle=True)
        scores = cross_val_score(regr, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
        return scores.mean()

    print('Optimizing model hyperparameters...')
    print('> Goal: Maximize neg MSE')
    start_time = time.time()
    # Optimize hyperparameters using Optuna
    try:
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=100, show_progress_bar=True, n_jobs=-1)
    except Exception as e:
        print(f"Error during optimization: {e}")
    elapsed_time = (time.time() - start_time)
    statsTrain['OptimTime'].append(elapsed_time)

    print('\nOptimized parameters:')
    for key, value in study.best_params.items():
        print(f'  {key}\t{value}')

# Train the best model
    best_model = MultiOutputRegressor(
        estimator=HistGradientBoostingRegressor(
            max_depth=study.best_params['max_depth'],
            min_samples_leaf=study.best_params['min_samples_leaf'],
            learning_rate=study.best_params['learning_rate'],
            max_iter=study.best_params['max_iter'],
        )
    )

    start_time = time.time()
    print('\nFitting model with optimized hyperparameters...')
    best_model.fit(X_train, y_train)
    elapsed_time = (time.time() - start_time)
    print("Training time %.3f s" % elapsed_time)

# Compute training stats
    y_pred = best_model.predict(X_train)
    models.append(name)

    r2, RMSE, MAE, MedAE, MSE = computeStats(y_train, y_pred)
    statsTrain['R2'].append(r2)
    statsTrain['RMSE'].append(RMSE)
    statsTrain['MAE'].append(MAE)
    statsTrain['MedAE'].append(MedAE)
    statsTrain['MSE'].append(MSE)
    statsTrain['Time'].append(elapsed_time)

# Save training predictions
    saveXLSX(pd.DataFrame(y_pred), name+'_train_ypred', subfolder='GBR')

# Compute test stats
    y_pred = best_model.predict(X_test)

    r2, RMSE, MAE, MedAE, MSE = computeStats(y_test, y_pred)
    statsTest['R2'].append(r2)
    statsTest['RMSE'].append(RMSE)
    statsTest['MAE'].append(MAE)
    statsTest['MedAE'].append(MedAE)
    statsTest['MSE'].append(MSE)

# Save test predictions
    saveXLSX(pd.DataFrame(y_pred), name+'_test_ypred', subfolder='GBR')

    return best_model

def MLP_model(X_train, X_test, y_train, y_test, name: str):
    print('\n::: ANN :::')

    def objective(trial):
# Define the hyperparameter search space
        alpha = trial.suggest_categorical('alpha', [1E-5, 1E-4, 1E-3, 1E-2, 1E-1])
        hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [8, 16, 32, 64])
        activation = trial.suggest_categorical('activation', ['logistic', 'relu', 'tanh'])
        learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
        early_stopping = trial.suggest_categorical('early_stopping', [True, False])

# Define the model
        regr = MLPRegressor(
            alpha=alpha,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver='adam',
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            max_iter=10000
        )

# Perform cross-validation
        kfold = KFold(n_splits=kval, shuffle=True)
        scores = cross_val_score(regr, X_train, y_train, cv=kfold, scoring='r2', n_jobs=-1)
        return scores.mean()

    print('Optimizing model hyperparameters...')
    print('> Goal: Maximize R2')
    start_time = time.time()
# Optimize hyperparameters using Optuna
    try:
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=1000, show_progress_bar=True, n_jobs=1)
    except Exception as e:
        print(f"Error during optimization: {e}")
    elapsed_time = (time.time() - start_time)
    statsTrain['OptimTime'].append(elapsed_time)

    print('\nOptimized parameters:')

    for key, value in study.best_params.items():
        print(f'  {key}\t{value}')

# Train the best model
    # hidden_layer_sizes=2**study.best_params['exponent']
    best_model = MLPRegressor(
        alpha=study.best_params['alpha'],
        hidden_layer_sizes=study.best_params['hidden_layer_sizes'],
        activation=study.best_params['activation'],
        solver='adam',
        learning_rate=study.best_params['learning_rate'],
        early_stopping=study.best_params['early_stopping'],
        max_iter=10000
    )

    start_time = time.time()
    print('\nFitting model with optimized hyperparameters...')
    best_model.fit(X_train, y_train)
    elapsed_time = (time.time() - start_time)
    print("Training time %.3f s" % elapsed_time)

# Compute training stats
    y_pred = best_model.predict(X_train)
    models.append(name)

    r2, RMSE, MAE, MedAE, MSE = computeStats(y_train, y_pred)
    statsTrain['R2'].append(r2)
    statsTrain['RMSE'].append(RMSE)
    statsTrain['MAE'].append(MAE)
    statsTrain['MedAE'].append(MedAE)
    statsTrain['MSE'].append(MSE)
    statsTrain['Time'].append(elapsed_time)

# Save training predictions
    saveXLSX(pd.DataFrame(y_pred), name+'_train_ypred', subfolder='ANN')

# Compute test stats
    y_pred = best_model.predict(X_test)

    r2, RMSE, MAE, MedAE, MSE = computeStats(y_test, y_pred)
    statsTest['R2'].append(r2)
    statsTest['RMSE'].append(RMSE)
    statsTest['MAE'].append(MAE)
    statsTest['MedAE'].append(MedAE)
    statsTest['MSE'].append(MSE)

# Save test predictions
    saveXLSX(pd.DataFrame(y_pred), name+'_test_ypred', subfolder='ANN')

    return best_model

# PAIRWISE DIFFERENCE MODELS
def PRF_model(X_train, X_test, y_train, y_test, name: str):
    print('\n::: PAIRWISE DIF RF :::')
# optuna.logging.set_verbosity(optuna.logging.INFO)

# Prepare expanded data
    expTrain_X = genFeatures(X_train,X_train)
    expTrain_y = genTargets(y_train, y_train)

    def objective(trial):
# Define the hyperparameter search space
        n_estimators = trial.suggest_int('n_estimators', 100, 500)
        max_depth = trial.suggest_int('max_depth', 10, 32, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)

        regr = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=True,
            max_features='sqrt'
        )

        kfold = KFold(n_splits=kval, shuffle=True)
        scores = cross_val_score(regr, expTrain_X, expTrain_y, cv=kfold, scoring='neg_mean_squared_error')
        return scores.mean()

    print('\nOptimizing model hyperparameters...')
    print('> Goal: Maximize neg MSE')
    start_time = time.time()
    try:
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=20, show_progress_bar=True, n_jobs=-1) # Reduced to 10 trials
    except Exception as e:
        print(f"Error during optimization: {e}")
    elapsed_time = (time.time() - start_time)
    statsTrain['OptimTime'].append(elapsed_time)

    print('\nOptimized parameters:')
    for key, value in study.best_params.items():
        print(f'  {key}\t{value}')

    best_model = RandomForestRegressor(
        n_estimators=study.best_params['n_estimators'],
        max_depth=study.best_params['max_depth'],
        min_samples_split=study.best_params['min_samples_split'],
        min_samples_leaf=study.best_params['min_samples_leaf'],
        max_features='sqrt'
    )
    start_time = time.time()
    print('\nFitting model with optimized hyperparameters...')
    best_model.fit(expTrain_X, expTrain_y)
    elapsed_time = (time.time() - start_time)
    print("Training time %.3f s" % elapsed_time)

    expTest_X  = genFeatures(X_train, X_train)
    y_test_minus_y_pred = best_model.predict(expTest_X)

    y_pred_mu, y_pred_std = twinPredictorHelper(X_train, X_train, y_train, y_test_minus_y_pred)
    models.append(name)

    r2, RMSE, MAE, MedAE, MSE = computeStats(y_train, y_pred_mu)
    statsTrain['R2'].append(r2)
    statsTrain['RMSE'].append(RMSE)
    statsTrain['MAE'].append(MAE)
    statsTrain['MedAE'].append(MedAE)
    statsTrain['MSE'].append(MSE)
    statsTrain['Time'].append(elapsed_time)

    saveXLSX(pd.DataFrame(y_pred_mu), name+'_train_ypred_mu', subfolder='PD-RF')
    saveXLSX(pd.DataFrame(y_pred_std), name+'_train_ypred_std', subfolder='PD-RF')

    print('\n* EXPANDED TESTING DATA *')
    expTest_X  = genFeatures(X_test, X_train)
    y_test_minus_y_pred = best_model.predict(expTest_X)

    y_pred_mu, y_pred_std = twinPredictorHelper(X_train, X_test, y_train, y_test_minus_y_pred)

    r2, RMSE, MAE, MedAE, MSE = computeStats(y_test, y_pred_mu)
    statsTest['R2'].append(r2)
    statsTest['RMSE'].append(RMSE)
    statsTest['MAE'].append(MAE)
    statsTest['MedAE'].append(MedAE)
    statsTest['MSE'].append(MSE)

    saveXLSX(pd.DataFrame(y_pred_mu), name+'_test_ypred_mu', subfolder='PD-RF')
    saveXLSX(pd.DataFrame(y_pred_std), name+'_test_ypred_std', subfolder='PD-RF')

    return best_model, y_pred_mu, y_pred_std

def PGBR_model(X_train, X_test, y_train, y_test, name: str):
    print('\n::: PAIRWISE DIF GBR :::')

    expTrain_X = genFeatures(X_train,X_train)
    expTrain_y = genTargets(y_train, y_train)

    def objective(trial):
        max_iter = trial.suggest_int('max_iter', 50, 500)
        max_depth = trial.suggest_int('max_depth', 5, 15)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 10, 100)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        max_bins = trial.suggest_int('max_bins', 128, 255)
        l2_regularization = trial.suggest_float('l2_regularization', 1E-6, 10, log=True)
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 50, 500)

        regr = MultiOutputRegressor(
            estimator=HistGradientBoostingRegressor(
                max_iter=max_iter,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                learning_rate=learning_rate,
                max_bins=max_bins,
                l2_regularization=l2_regularization,
                max_leaf_nodes=max_leaf_nodes,
                early_stopping=True,
                n_iter_no_change=10,
                tol=1e-2,
            )
        )

        kfold = KFold(n_splits=kval, shuffle=True)
        scores = cross_val_score(regr, expTrain_X, expTrain_y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
        return scores.mean()

    print('Optimizing model hyperparameters...')
    print('> Goal: Maximize neg MSE')
    start_time = time.time()
    try:
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=25, show_progress_bar=True, n_jobs=1) # Reduced to 10 trials
    except Exception as e:
        print(f"Error during optimization: {e}")
    elapsed_time = (time.time() - start_time)
    statsTrain['OptimTime'].append(elapsed_time)

    print('\nOptimized parameters:')
    for key, value in study.best_params.items():
        print(f'  {key}\t{value}')

    best_model = MultiOutputRegressor(
        estimator=HistGradientBoostingRegressor(
            max_depth=study.best_params['max_depth'],
            min_samples_leaf=study.best_params['min_samples_leaf'],
            learning_rate=study.best_params['learning_rate'],
            max_iter=study.best_params['max_iter']
        )
    )

    start_time = time.time()
    print('\nFitting model with optimized hyperparameters...')
    best_model.fit(expTrain_X, expTrain_y)
    elapsed_time = (time.time() - start_time)
    print("Training time %.3f s" % elapsed_time)

    expTest_X  = genFeatures(X_train, X_train)
    y_test_minus_y_pred = best_model.predict(expTest_X)

    y_pred_mu, y_pred_std = twinPredictorHelper(X_train, X_train, y_train, y_test_minus_y_pred)
    models.append(name)

    r2, RMSE, MAE, MedAE, MSE = computeStats(y_train, y_pred_mu)
    statsTrain['R2'].append(r2)
    statsTrain['RMSE'].append(RMSE)
    statsTrain['MAE'].append(MAE)
    statsTrain['MedAE'].append(MedAE)
    statsTrain['MSE'].append(MSE)
    statsTrain['Time'].append(elapsed_time)

    saveXLSX(pd.DataFrame(y_pred_mu), name+'_train_ypred_mu', subfolder='PD-GBR')
    saveXLSX(pd.DataFrame(y_pred_std), name+'_train_ypred_std', subfolder='PD-GBR')

    print('\n* EXPANDED TESTING DATA *')
    expTest_X  = genFeatures(X_test, X_train)
    y_test_minus_y_pred = best_model.predict(expTest_X)

    y_pred_mu, y_pred_std = twinPredictorHelper(X_train, X_test, y_train, y_test_minus_y_pred)

    r2, RMSE, MAE, MedAE, MSE = computeStats(y_test, y_pred_mu)
    statsTest['R2'].append(r2)
    statsTest['RMSE'].append(RMSE)
    statsTest['MAE'].append(MAE)
    statsTest['MedAE'].append(MedAE)
    statsTest['MSE'].append(MSE)

    saveXLSX(pd.DataFrame(y_pred_mu), name+'_test_ypred_mu', subfolder='PD-GBR')
    saveXLSX(pd.DataFrame(y_pred_std), name+'_test_ypred_std', subfolder='PD-GBR')

    return best_model, y_pred_mu, y_pred_std

def PNN_model(X_train, X_test, y_train, y_test, name: str):
    print('\n::: PAIRWISE DIF ANN :::')

    expTrain_X = genFeatures(X_train,X_train)
    expTrain_y = genTargets(y_train, y_train)

    def objective(trial):
        alpha = trial.suggest_categorical('alpha', [1E-5, 1E-4, 1E-3, 1E-2, 1E-1])
        hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [16, 32, 64, 128, 256])
        activation = trial.suggest_categorical('activation', ['logistic', 'relu', 'tanh'])
        learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
        early_stopping = trial.suggest_categorical('early_stopping', [True, False])

        regr = MLPRegressor(
            alpha=alpha,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver='adam',
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            max_iter=1000
        )

        kfold = KFold(n_splits=kval, shuffle=True)
        scores = cross_val_score(regr, expTrain_X, expTrain_y, cv=kfold, scoring='r2', n_jobs=-1)
        return scores.mean()

    print('\nOptimizing model hyperparameters...')
    print('> Goal: Maximize R2')

    start_time = time.time()
    try:
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=25, show_progress_bar=True, n_jobs=1)
    except Exception as e:
        print(f"Error during optimization: {e}")
    elapsed_time = (time.time() - start_time)
    statsTrain['OptimTime'].append(elapsed_time)
    
    print('\nOptimized parameters:')
    for key, value in study.best_params.items():
        print(f'  {key}\t{value}')

    best_model = MLPRegressor(
        alpha=study.best_params['alpha'],
        hidden_layer_sizes=study.best_params['hidden_layer_sizes'],
        activation=study.best_params['activation'],
        solver='adam',
        learning_rate=study.best_params['learning_rate'],
        early_stopping=study.best_params['early_stopping'],
        max_iter=10000
    )

    print('\nStarted PNN training')
    start_time = time.time()
    best_model.fit(expTrain_X,expTrain_y)
    
    elapsed_time = time.time() - start_time
    print("Training time %.3f seconds" % elapsed_time)

    expTest_X  = genFeatures(X_train, X_train)
    y_test_minus_y_pred = best_model.predict(expTest_X)

    y_pred_mu, y_pred_std = twinPredictorHelper(X_train, X_train, y_train, y_test_minus_y_pred)
    models.append(name)

    r2, RMSE, MAE, MedAE, MSE = computeStats(y_train, y_pred_mu)
    statsTrain['R2'].append(r2)
    statsTrain['RMSE'].append(RMSE)
    statsTrain['MAE'].append(MAE)
    statsTrain['MedAE'].append(MedAE)
    statsTrain['MSE'].append(MSE)
    statsTrain['Time'].append(elapsed_time)

    saveXLSX(pd.DataFrame(y_pred_mu), name+'_train_ypred_mu', subfolder='PD-ANN')
    saveXLSX(pd.DataFrame(y_pred_std), name+'_train_ypred', subfolder='PD-ANN')

    print('\n* EXPANDED TESTING DATA *')
    expTest_X  = genFeatures(X_test, X_train)
    y_test_minus_y_pred = best_model.predict(expTest_X)

    y_pred_mu, y_pred_std = twinPredictorHelper(X_train, X_test, y_train, y_test_minus_y_pred)

    r2, RMSE, MAE, MedAE, MSE = computeStats(y_test, y_pred_mu)
    statsTest['R2'].append(r2)
    statsTest['RMSE'].append(RMSE)
    statsTest['MAE'].append(MAE)
    statsTest['MedAE'].append(MedAE)
    statsTest['MSE'].append(MSE)

    saveXLSX(pd.DataFrame(y_pred_mu), name+'_test_ypred_mu', subfolder='PD-ANN')
    saveXLSX(pd.DataFrame(y_pred_std), name+'_test_ypred_std', subfolder='PD-ANN')

    return best_model, y_pred_mu, y_pred_std

optuna.logging.set_verbosity(optuna.logging.WARNING)

columns = ['inputNi', 'inputMn', 'inputCo',
           'temp', 'pKa1', 'pKa2', 'pKa3',
           'nProtons', 'sLi', 'sNi', 'sMn', 'sCo',
           'acidC', 'H2O2_conc', 'solidToLiquid','time']

X, y = importdata(dataPath, columns)

X_train, X_test, y_train, y_test = splitData(X, y)
X_train_filledNaN = X_train.fillna(X_train.median())
X_test_filledNaN = X_test.fillna(X_train.median())

model_configurations = [
    {
        'name': 'RF_full',
        'model_func': RF_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'save_path': '',
        'drop_cols': None,
        'comments': 'RF using all features.',
        'subfolder': 'RF'
    },
    {
        'name': 'GBR_full',
        'model_func': GBR_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'save_path': '',
        'drop_cols': None,
        'comments': 'GBR using all features.',
        'subfolder': 'GBR'
    },
    {
        'name': 'ANN_full',
        'model_func': MLP_model,
        'X_train': X_train_filledNaN,
        'X_test': X_test_filledNaN,
        'y_train': y_train,
        'y_test': y_test,
        'save_path': '',
        'drop_cols': None,
        'comments': "ANN with missing values filled with median \n(ANN don't support NaN values)",
        "subfolder": "ANN"
    },
    {
        'name': 'ANN_no_pKa_sol',
        'model_func': MLP_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'save_path': '',
        'drop_cols': ['pKa2', 'pKa3', 'sLi', 'sNi', 'sMn', 'sCo'],
        'comments': "ANN dropping extra pKas, solubilities \n(ANN don't support NaN values)",
        "subfolder": "ANN"
    },
    {   
        'name': 'PD-ANN_full',
        'model_func': PNN_model,
        'X_train': X_train_filledNaN,
        'X_test': X_test_filledNaN,
        'y_train': y_train,
        'y_test': y_test,
        'save_path': '',
        'drop_cols': None,
        'comments': "PD-ANN with missing values filled with median \n(ANN don't support NaN values)",
        'subfolder': 'PD-ANN'
    },
    {   
        'name': 'PD-ANN_no_pKa_sol',
        'model_func': PNN_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'save_path': '',
        'drop_cols': ['pKa2', 'pKa3', 'sLi', 'sNi', 'sMn', 'sCo'],
        'comments': "PD-ANN dropping extra pKas, solubilities \n(ANN don't support NaN values)",
        'subfolder': 'PD-ANN'
    },
    {
        'name': 'PD-GBR_full',
        'model_func': PGBR_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'save_path': '',
        'drop_cols': None,
        'comments': 'PD-GBR using all features.',
        'subfolder': 'PD-GBR'
    },
    {
        'name': 'PD-GBR_no_pKa',
        'model_func': PGBR_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'save_path': '',
        'drop_cols': ['pKa2', 'pKa3'],
        'comments': 'PD-GBR dropping pKa2, pKa3',
        'subfolder': 'PD-GBR'
    },
    {
        'name': 'PD-GBR_no_sol',
        'model_func': PGBR_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'save_path': '',
        'drop_cols': ['sLi', 'sNi', 'sMn', 'sCo'],
        'comments': 'PD-GBR dropping solubilities',
        'subfolder': 'PD-GBR'
    },
    {
        'name': 'PD-GBR_no_lipinski',
        'model_func': PGBR_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'save_path': '',
        'drop_cols': ['logP', 'hba', 'hbd', 'Mr_acid', 'tpsa'],
        'comments': 'PD-GBR dropping logP, HBA, HBD, MW, TPSA',
        'subfolder': 'PD-GBR'
    },
    {
        'name': 'PD-GBR_no_charge_pol',
        'model_func': PGBR_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'save_path': '',
        'drop_cols': ['max_partial_charge', 'min_partial_charge', 'molar_refractivity'],
        'comments': 'PD-GBR dropping max/min partial charge and molar refractivity',
        'subfolder': 'PD-GBR'
    },
    {
        'name': 'PD-GBR_no_FG',
        'model_func': PGBR_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'save_path': '',
        'drop_cols': ['fr_COO', 'fr_C_O', 'fr_NH0', 'fr_NH2', 'fr_halogen', 'fr_Al_OH', 'fr_S'],
        'comments': 'PD-GBR dropping functional groups \n(COO, C_O, NH0, NH2, halogen, Al_OH, S)',
        'subfolder': 'PD-GBR'
    }
]

trained_models = {}

for config in model_configurations:
    print("\n")
    print("="*50)  # Add a separator line
    print(f"Training model: {config['name']}")
    print(f"Description: {config['comments']}")  # Print the comments field
    # Print the starting time
    print(f"Starting time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)  # Add a separator line

    X_train_model = config['X_train']
    X_test_model = config['X_test']

    if config['drop_cols']:
        X_train_model = X_train_model.drop(columns=config['drop_cols'])
        X_test_model = X_test_model.drop(columns=config['drop_cols'])

        saveModel(X_train_model, f"xtrain_{config['name']}", subfolder=config['subfolder'])
        saveModel(X_test_model, f"xtest_{config['name']}", subfolder=config['subfolder'])

        saveXLSX(X_train_model, f"xtrain_{config['name']}", subfolder=config['subfolder'])
        saveXLSX(X_test_model, f"xtest_{config['name']}", subfolder=config['subfolder'])

    if config['model_func'] in [PRF_model, PGBR_model, PNN_model]:
        trained_model, _, _ = config['model_func'](X_train_model, X_test_model, config['y_train'], config['y_test'], config['name'])   
    else:
        trained_model = config['model_func'](X_train_model, X_test_model, config['y_train'], config['y_test'], config['name'])

    saveModel(trained_model, config['name'], subfolder=config['subfolder'])
    trained_models[config['name']] = trained_model

saveModel(X_train, 'xtrain_full', subfolder=None)
saveModel(y_train, 'ytrain_full', subfolder=None)
saveModel(X_test, 'xtest_full', subfolder=None)
saveModel(y_test, 'ytest_full', subfolder=None)

saveXLSX(X_train, 'xtrain_full', subfolder=None)
saveXLSX(y_train, 'ytrain_full', subfolder=None)
saveXLSX(X_test, 'xtest_full', subfolder=None)
saveXLSX(y_test, 'ytest_full', subfolder=None)

writer = pd.ExcelWriter(os.path.join(output_dir, 'training_stats.xlsx'))
stats = pd.DataFrame(statsTrain, index=models)
stats.to_excel(writer, sheet_name='Training')
print('\n::: TRAINING STATS :::')
print(stats)

stats = pd.DataFrame(statsTest, index=models)
stats.to_excel(writer, sheet_name='Test')

# Ensure all sheets are visible before closing the writer
for sheet in writer.sheets.values():
    sheet.sheet_state = 'visible'

writer.close()

print('\n::: TEST STATS :::')
print(stats)

def savePreds(writer, y_pred, name):
    if not isinstance(y_pred, pd.DataFrame):
        y_pred = pd.DataFrame(y_pred, columns=['xLi', 'xNi', 'xMn', 'xCo'])
    y_pred.to_excel(writer, sheet_name=name)

# Make predictions on case study 2
'''
print('\n'+"="*50)
print("Making predictions on case study 2...")
X_CS2,_ = importdata('data/CaseStudy2.xlsx', columns)
X_CS2_filledNaN = X_CS2.fillna(X_CS2.median())

cs2_output_dir = os.path.join(output_dir, 'CS2')
os.makedirs(cs2_output_dir, exist_ok=True)

def predict_and_save(model, X, name, twin=False, subfolder=None, c_drop=None, use_NaN=False):
    if use_NaN:
        # If the model is trained with NaN values, use the correct X_train
        X_train_current = X_train_filledNaN
    else:
        # If it's a regular model (i.e., not a MLP)
        X_train_current = X_train

    if c_drop:
        # Drop the specified columns from X and X_train
        X = X.drop(columns=c_drop)
        X_train_current = X_train_current.drop(columns=c_drop)

    if twin:
        expanded_X = genFeatures(X, X_train_current)
        y_CS2_minus_y_pred = model.predict(expanded_X)
        y_CS2_mu, y_CS2_std = twinPredictorHelper(X_train_current, X, y_train, y_CS2_minus_y_pred)
        y_CS2_mu = pd.DataFrame(y_CS2_mu, columns=['xLi', 'xNi', 'xMn', 'xCo'])
        y_CS2_std = pd.DataFrame(y_CS2_std, columns=['dLi', 'dNi', 'dMn', 'dCo'])
        y_CS2 = pd.concat([y_CS2_mu, y_CS2_std], axis=1)
        saveXLSX(y_CS2, name, subfolder=os.path.join('CS2', subfolder) if subfolder else 'CS2')
    else:
        y_CS2_mu = model.predict(X)
        y_CS2 = pd.DataFrame(y_CS2_mu, columns=['xLi', 'xNi', 'xMn', 'xCo'])
        saveXLSX(y_CS2, name, subfolder=os.path.join('CS2', subfolder) if subfolder else 'CS2')

print("Making predictions with full feature set...")
predict_and_save(trained_models['RF_full'], X_CS2, 'RF_full', subfolder='RF')
predict_and_save(trained_models['GBR_full'], X_CS2, 'GBR_full', subfolder='GBR')
predict_and_save(trained_models['PD-GBR_full'], X_CS2, 'PD-GBR_full', twin=True, subfolder='PD-GBR')
predict_and_save(trained_models['ANN_full'], X_CS2_filledNaN, 'ANN_full', subfolder='ANN', use_NaN=True)
predict_and_save(trained_models['PD-ANN_full'], X_CS2_filledNaN, 'PD-ANN_full', twin=True, subfolder='PD-ANN', use_NaN=True)

print("Making predictions with trimmed feature set...")
c_drop = ['pKa2', 'pKa3', 'sLi', 'sNi', 'sMn', 'sCo']
predict_and_save(trained_models['ANN_no_pKa_sol'], X_CS2_filledNaN, 'ANN_no_pKa_sol', subfolder='ANN', c_drop=c_drop, use_NaN=True)
predict_and_save(trained_models['PD-ANN_no_pKa_sol'], X_CS2_filledNaN, 'PD-ANN_no_pKa_sol', twin=True, subfolder='PD-ANN', c_drop=c_drop, use_NaN=True)

print("Making predictions with trimmed feature set, no pKa...")
c_drop = ['pKa2', 'pKa3']
predict_and_save(trained_models['PD-GBR_no_pKa'], X_CS2, 'PD-GBR_no_pKa', twin=True, subfolder='PD-GBR', c_drop=c_drop)

print("Making predictions with trimmed feature set, no solubilities...")
c_drop = ['sLi', 'sNi', 'sMn', 'sCo']
predict_and_save(trained_models['PD-GBR_no_sol'], X_CS2, 'PD-GBR_no_sol', twin=True, subfolder='PD-GBR', c_drop=c_drop)

print("Making predictions with trimmed feature set, no Lipinski descriptors...")
c_drop = ['logP', 'hba', 'hbd', 'Mr_acid', 'tpsa']
predict_and_save(trained_models['PD-GBR_no_lipinski'], X_CS2, 'PD-GBR_no_lipinski', twin=True, subfolder='PD-GBR', c_drop=c_drop)

print("Making predictions with trimmed feature set, no charge and polarizability...")
c_drop = ['max_partial_charge', 'min_partial_charge', 'molar_refractivity']
predict_and_save(trained_models['PD-GBR_no_charge_pol'], X_CS2, 'PD-GBR_no_charge_pol', twin=True, subfolder='PD-GBR', c_drop=c_drop)

print("Making predictions with trimmed feature set, no functional groups...")
c_drop = ['fr_COO', 'fr_C_O', 'fr_NH0', 'fr_NH2', 'fr_halogen', 'fr_Al_OH', 'fr_S']
predict_and_save(trained_models['PD-GBR_no_FG'], X_CS2, 'PD-GBR_no_FG', twin=True, subfolder='PD-GBR', c_drop=c_drop)
'''

# Restore console output at the end of the script if redirected
if redirect_output:
    sys.stdout.file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__