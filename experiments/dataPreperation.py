import pandas as pd

eccd = pd.read_csv('./data/eccd/eccd.csv')
eccd = eccd.rename(columns={'Class': 'Target'})

wdbc = pd.read_csv('./data/wdbc/wdbc.data', header=None)
wdbc.columns = ['ID', 'Diagnosis', 'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness', 'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension', 'SE Radius', 'SE Texture', 'SE Perimeter', 'SE Area', 'SE Smoothness', 'SE Compactness', 'SE Concavity', 'SE Concave Points', 'SE Symmetry', 'SE Fractal Dimension', 'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area', 'Worst Smoothness', 'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension']
wdbc = wdbc.drop(columns=['ID'])
wdbc = wdbc.rename(columns={'Diagnosis': 'Target'})
wdbc['Target'] = wdbc['Target'].map({'M': 1, 'B': 0})

print(eccd.head())
print(wdbc.head())

eccd.to_pickle('./data/eccd/eccd.pkl')
wdbc.to_pickle('./data/wdbc/wdbc.pkl')