import numpy as np
import pandas as pd
import itertools
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

eccd = pd.read_csv('./data/eccd/eccd.csv')
eccd = eccd.rename(columns={'Class': 'Target'})

ieeecis = pd.read_csv('./data/ieeecis/ieeecis.csv')
ieeecis = ieeecis.drop(columns=['TransactionID'])
v_ohe = OneHotEncoder()
ieeecis_ohe = v_ohe.fit_transform(ieeecis[[x for x in ieeecis.columns if ieeecis[x].dtype not in [float, int]]]).toarray()
ieeecis_ohe = pd.DataFrame(ieeecis_ohe, columns= ["O{}".format(x) for x in range(ieeecis_ohe.shape[1])])
ieeecis_v = pd.concat([ieeecis[[x for x in ieeecis.columns if ieeecis[x].dtype in [float, int]]], ieeecis_ohe], axis=1)
ieeecis_v = ieeecis_v[[x for x in ieeecis_v.columns if x not in ["isFraud", "TransactionDT", "TransactionAmt"]]]
ieeecis_v = ieeecis_v.fillna(-1)
v_ss = StandardScaler()
ieeecis_v = pd.DataFrame(v_ss.fit_transform(ieeecis_v))
v_pca = PCA(n_components=28)
ieeecis_v_pca = pd.DataFrame(v_pca.fit_transform(ieeecis_v), columns= ["V" + str(x) for x in range(1, 29)])
ieeecis = pd.concat([ieeecis[["TransactionDT"]], ieeecis[["TransactionAmt"]], ieeecis[["isFraud"]], pd.DataFrame(ieeecis_v_pca)], axis=1)
ieeecis = ieeecis.rename(columns={'isFraud': 'Target'})

wdbc = pd.read_csv('./data/wdbc/wdbc.data', header=None)
wdbc.columns = ['ID', 'Diagnosis', 'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness', 'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension', 'SE Radius', 'SE Texture', 'SE Perimeter', 'SE Area', 'SE Smoothness', 'SE Compactness', 'SE Concavity', 'SE Concave Points', 'SE Symmetry', 'SE Fractal Dimension', 'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area', 'Worst Smoothness', 'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension']
wdbc = wdbc.drop(columns=['ID'])
wdbc = wdbc.rename(columns={'Diagnosis': 'Target'})
wdbc['Target'] = wdbc['Target'].map({'M': 1, 'B': 0})

#wpbc = pd.read_csv('./data/wpbc/wpbc.data', header=None)
#wpbc.columns = ['ID', 'Outcome', 'Time'] + [r[1] + "_" + r[0] for r in itertools.product(['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension'],['Mean', 'Std', 'Max'])] + ['Tumor Size', 'Lymph Node Status']
#wpbc = wpbc.drop(columns=['ID'])
#wpbc = wpbc.rename(columns={'Outcome': 'Target'})
#wpbc = wpbc.loc[wpbc["Lymph Node Status"] != "?"]
#wpbc["Lymph Node Status"] = wpbc["Lymph Node Status"].astype(int)
#wpbc['Target'] = wpbc['Target'].map({'R': 1, 'N': 0})

wbcd = pd.read_csv('./data/wbcd/breast-cancer-wisconsin.data', header=None)
wbcd.columns = ['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Diagnosis']
wbcd = wbcd.drop(columns=['ID'])
wbcd = wbcd.rename(columns={'Diagnosis': 'Target'})
wbcd = wbcd.replace("?", np.nan).dropna()
wbcd['Target'] = wbcd['Target'].map({4: 1, 2: 0})

print(eccd.head())
print(ieeecis.head())
print(wdbc.head())
print(wbcd.head())

eccd = eccd[eccd.columns].apply(pd.to_numeric, errors='coerce')
ieeecis = ieeecis[ieeecis.columns].apply(pd.to_numeric, errors='coerce')
wdbc = wdbc[wdbc.columns].apply(pd.to_numeric, errors='coerce')
wbcd = wbcd[wbcd.columns].apply(pd.to_numeric, errors='coerce')

eccd.to_pickle('./data/eccd/eccd.pkl')
wdbc.to_pickle('./data/wdbc/wdbc.pkl')
ieeecis.to_pickle('./data/ieeecis/ieeecis.pkl')
wbcd.to_pickle('./data/wbcd/wbcd.pkl')