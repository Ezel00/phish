
import numpy as np
import pandas as pd

#to get the stats from the txt files and form them into workable dfs
p1= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\embed\PCA12\phish\Ripser_20iter_50perc_random.txt"
p2 =r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\embed\PCA12\notPhish\Ripser_20iter_50perc_random.txt"
p1T= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\embed\PCA12Test\phish\Ripser_20iter_50perc_random.txt"
p2T =r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\embed\PCA12Test\notPhish\Ripser_20iter_50perc_random.txt"
p3=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\notEmbed\PCA12\notPhish\Ripser_20iter_50perc_random.txt"
p4=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\notEmbed\PCA12\phish\Ripser_20iter_50perc_random.txt"
p3T=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\notEmbed\PCA12Test\notPhish\Ripser_20iter_50perc_random.txt"
p4T=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\notEmbed\PCA12Test\phish\Ripser_20iter_50perc_random.txt"

def processStats(p1, p2):
    df1 = pd.read_csv(p1, header=None)
    df2 = pd.read_csv(p2, header=None)
    df1["label"] = 1
    df2["label"] = 0
    df= pd.concat([df1, df2], ignore_index=True)
    return df.sample(frac=1).reset_index(drop=True)



##not embeddings
train = processStats(p3, p4)  #pd.read_csv(pathTrain)
test =  processStats(p3T, p4T)  #pd.read_csv(pathTest)

XTrain = train.drop(columns="label")
yTrain = train["label"]
XTest =test.drop(columns="label")
yTest = test["label"]

##embedding
trainEmbed = processStats(p1, p2)  #pd.read_csv(pathTrain)
testEmbed =  processStats(p1T, p2T)  #pd.read_csv(pathTest)

XTrainEmbed = train.drop(columns="label")
yTrainEmbed = train["label"]
XTestEmbed =test.drop(columns="label")
yTestEmbed = test["label"]
