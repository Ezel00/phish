from itertools import combinations
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ripser import ripser, Rips
from persim import plot_diagrams, PersistenceImager, bottleneck
from skimage.transform import resize
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

#HERE, WE USE PERSISTANCE IMAGES TO TRAIN ML MODELS
# TRIED GETTING THE PERSISTANCE IMAGES BASED ON THE PAIRWSIE DISTANCE MATRICES OF
#20 FEATURES, PCA REDUCED FEATURES, AS WELL AS TDIF WORD EMBEDDINGS
pathTrain2= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\statDatasets\trainBarcodeStats.csv"
p3= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\statDatasets\trainEmbedBarcodeStats.csv"
df = (pd.read_csv(p3)).to_numpy()
condition = df[:, -1] > 0
#clean is phish
dataPhish =   df[condition]
dataNP =   df[~condition]

dataPhish =   dataPhish[:1500]
dataPhish = dataPhish[:, :-1]
dataNP =   dataNP[:1500]
dataNP = dataNP[:, :-1]
'''conditionT = dfT[:, -1] > 0
testP= dfT[conditionT]
testNP =   dfT[~conditionT]
dataPT = testP[:, :-1]
dataNPT = testNP[:, :-1]
testLabels = [0 for _ in range(len(dataNPT))] + [1 for _ in range(len(dataPT))]'''

#these are arrays of len = #datapoints for each
DnotP = pairwise_distances(dataNP, metric='euclidean')
DP = pairwise_distances(dataPhish, metric='euclidean')
'''DnotPT = pairwise_distances(dataNPT, metric='euclidean')
DPT = pairwise_distances(dataPT, metric='euclidean')'''

np.save("DnotP.npy", DnotP)
np.save("DP.npy", DP)
labels = [0  for _ in dataPhish] + [1 for _ in dataPhish]
#labelsT = [0  for _ in dataPT] + [1 for _ in dataNPT]

def splFurther(lst):
    def split_into_pairs(lst):
        pairs = [[lst[i], lst[i + 1]] for i in range(0, len(lst) - 1, 2)]
        if len(lst) % 2 != 0:
            pairs.append([lst[-1], lst[-1]])
        return pairs
    l = []
    for smallL in lst:
        l.append(split_into_pairs(smallL))
    return l

def getImgArray(DnotP, DP):
    rips = Rips(maxdim=1, coeff=2)
    DnotP = splFurther(DnotP)
    DP = splFurther(DP)
    datas = DnotP + DP
    diagrams = [rips.fit_transform(np.array(data)) for data in datas]
    diagrams_h1 = [dgm[1] for dgm in diagrams if len(dgm) > 1]  # Ensure H1 diagrams exist
    # Handle empty persistence diagrams
    if not diagrams_h1:
        raise ValueError("All persistence diagrams are empty!")
    pimgr = PersistenceImager()
    #pimgr.persistence_image_shape = (60, 60)
    pimgr.fit(diagrams_h1)
    imgs = pimgr.transform(diagrams_h1)
    # Resize images
    def resize_image(img, target_size=(30, 30)):
        if img is None or img.size == 0:
            return np.zeros(target_size, dtype=np.float32)  
        return resize(img, target_size, mode='reflect', anti_aliasing=True).astype(np.float32)
    imgs_array = np.array([resize_image(img).flatten() for img in imgs])
    return imgs_array


# convert the persistence images to a format
#the ml algos can use
def convertImages(DnotP, DP):
    imgs_array = getImgArray(DnotP, DP)
    plt.figure(figsize=(15, 7.5))
    X_train, X_test, y_train, y_test = train_test_split(
        imgs_array, labels, test_size=0.30, random_state=42
    )
    return X_train, X_test, y_train, y_test

def saveP(DnotP, DP):
    X_train, X_test, y_train, y_test = convertImages(DnotP, DP)
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    return 0

saveP(DnotP, DP)
