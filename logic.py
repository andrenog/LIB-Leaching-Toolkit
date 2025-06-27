import pandas as pd
import numpy as np
import joblib
import os

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen

# UTILITIES
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

def importdata(path, cols):
    print('Importing data from ', os.path.basename(path))

    # Load data using pandas
    trainData = pd.read_excel(path, header=0)

    # Separate features and targets
    try:
        X = trainData.loc[:, cols]
    except KeyError as e:
        missing_cols = set(cols) - set(trainData.columns)
        raise KeyError(f"Missing columns {missing_cols} in the imported data. "
                       f"Please ensure the file has the required columns: {cols}") from e

    y = trainData.loc[:, ['xLi', 'xNi', 'xMn', 'xCo']]

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
    # print('\n::: FEATURES :::')
    # print(X.head())
    print("- Features (X):\t", X.shape)
    # print("X data type:\t", type(X))

    # print('\n::: TARGETS :::')
    # print(y.head())
    print("- Targets  (y):\t", y.shape)
    # print("y data type:\t", type(y))

    return X, y


def genFeatures(X1, X2):
    # Closely follows the pseudocode in doi.org/10.1021/acs.jcim.1c00670
    print('- Expanding features')

    # Remove SMILES column if present
    if 'SMILES' in X1.columns:
        X1 = X1.drop(columns=['SMILES'])
    if 'SMILES' in X2.columns: 
        X2 = X2.drop(columns=['SMILES'])

    # Input arrays
    X1 = X1.to_numpy()
    X2 = X2.to_numpy()

    print("- Original X1:", X1.shape)
    print("- Original X2:", X2.shape)

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
    print("- Combined + reshaped X:", X1X2_combined.shape)
    # print(X1X2_combined.shape)
    # print(type(X1X2_combined))
    # print(X1X2_combined[:3])
    return X1X2_combined


def twinPredictorHelper(X_train, X_test, y_train, y_test_minus_y_pred):
    n1 = X_test.shape[0]
    n2 = X_train.shape[0]
    y_pred_distribution = np.zeros((n1, n2))
    y_pred_mu = np.zeros((X_test.shape[0], y_train.shape[1]))
    y_pred_std = np.zeros(y_pred_mu.shape)

    print('- Computing predictions')

    # iterate over the columns to compute predictions and stdev
    for i in range(y_test_minus_y_pred.shape[1]):
        currCol = y_test_minus_y_pred[:, i]
        currTrain = y_train.to_numpy()[:, i]

        # reshape the results to the appropriate format so that each row
        # corresponds to each element of the testing data, and each column
        # is the prediction anchored by the training data

        y_pred_distribution = currCol.reshape(n1, n2) + currTrain

        # Calculate the prediction as the average across anchored predictions
        y_pred_mu[:, i] = np.mean(y_pred_distribution, axis=1)

        # Calculate the deviation of the predictions
        y_pred_std[:, i] = np.std(y_pred_distribution, axis=1)

    return y_pred_mu, y_pred_std
