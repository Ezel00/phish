import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

testP = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\test.csv"
trainP=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\train.csv"
p1 =pd.read_csv(testP)
p2 =pd.read_csv(trainP)

#pass the data, number of components, binary rand var to represent train or test data
#true is train
def pcaDf(merged, num, train):
    scaler = StandardScaler()
    y = merged['phish']
    merged.columns = merged.columns.astype(str)
    X_scaled = scaler.fit_transform(merged)
    pca = PCA(n_components=num)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    cols = [f"component {i}" for i in range(num)]
    df_pca = pd.DataFrame(X_pca, columns=cols)
    test = ("Train" if train else "Test")
    op= fr"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet1\dfPCA{num}{test}.csv"
    df_pca["label"] = y.values
    print(df_pca.head())
    df_pca.to_csv(op, index=False)
    return df_pca

nums = [4,8,12]

#create PCA with 4,8,12 components
for num in nums:
    pcaDf(p1, num, train=False)
    pcaDf(p2, num, train=True)
