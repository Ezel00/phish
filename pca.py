import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

testP = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\test.csv"
trainP=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\train.csv"
p1 =pd.read_csv(testP)
p2 =pd.read_csv(trainP)

#we will also get the embeddings here and create a few PCA feature matrices from that
def readF(l2):
    with open(l2, "r", encoding="utf-8") as file:
        lis = file.read().splitlines()
    return lis
pathPhish= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\phishTrain.txt"
pathNP= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\notpTrain.txt"
pathPhishTest= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\phishTest.txt"
pathNPTest = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\notpTest.txt"
#paths to save the output csv for embeddings
pathEmbedTest=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\testEMBED.csv"
pathEmbedTrain=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\trainEMBED.csv"

phishL=readF(pathPhish)
notPL =readF(pathNP)
phishTestL=readF(pathPhishTest)
notPTestL=readF(pathNPTest)

#get the embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
Xembed = np.array(model.encode(phishL+notPL))
labels = [1 for _ in phishL] + [0 for _ in notPL]
print(len(phishL), len(notPL), labels[:5], labels[:-5])
XembedTest = np.array(model.encode(phishTestL+notPTestL))
labelsTest = [1 for _ in phishTestL] + [0 for _ in notPTestL]
#save the embeddings as a csv file
import numpy as np
import pandas as pd


def saveEmbed(embeddings, labels, filename):
    if len(embeddings) != len(labels):
        raise ValueError("Mismatch: embeddings and labels must have the same number of samples.")
    num_features = embeddings.shape[1]
    col_names = [f"feature_{i + 1}" for i in range(num_features)]
    df = pd.DataFrame(embeddings, columns=col_names)
    df["label"] = labels  # Add labels column
    df.to_csv(filename, index=False)
    print(f"Saved embeddings to {filename}")
    return df

saveEmbed(Xembed, labels, pathEmbedTrain)
saveEmbed(XembedTest, labelsTest, pathEmbedTest)
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
    df_pca.to_csv(op, index=False)
    return df_pca

def pcaDfEmbed(xs, labels, num, train):
    scaler = StandardScaler()
    if isinstance(xs, pd.DataFrame):
        X_scaled = scaler.fit_transform(xs.values)  # Convert DataFrame to NumPy array before scaling
    else:
        X_scaled = scaler.fit_transform(xs)  # Assume it's already a NumPy array
    pca = PCA(n_components=num)
    X_pca = pca.fit_transform(X_scaled)
    cols = [f"component {i + 1}" for i in range(num)]
    df_pca = pd.DataFrame(X_pca, columns=cols)
    df_pca["label"] = labels
    test_type = "Train" if train else "Test"
    output_path = fr"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet2\dfPCA{num}{test_type}.csv"
    df_pca.to_csv(output_path, index=False)
    print(df_pca.head())
    return df_pca

nums = [4,8,12]

#create PCA with 4,8,12 components
for num in nums:
    pcaDf(p1, num, train=False)
    pcaDf(p2, num, train=True)
