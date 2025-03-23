import pandas as pd
import matplotlib.pyplot as plt
import persim
from ripser import ripser
from sklearn.metrics import pairwise_distances
pathFull=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\train.csv"
path12=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet1\dfPCA12Train.csv"
pathEmbedFull=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\trainEMBED.csv"
pathEmbed12=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet2\dfPCA4Train.csv"
pathEmbedExtra = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\extrasEmbed\dfPCA260Train.csv"
df = pd.read_csv(pathEmbedExtra) 

#prep the data for visualization
phishes =  df[df['label'] == 1]
notP =  df[df['label'] == 0]
phishes = phishes.drop(columns=['label'])
notP = notP.drop(columns=['label'])


def plots(arrs, phish, num):
    DnotP = pairwise_distances(arrs, metric='cosine')
    diagrams = ripser(DnotP, metric="cosine", distance_matrix=True)['dgms']
    h1Color = 'pink' if phish else 'blue'
    colors = ['yellow', h1Color] 
    plt.figure(figsize=(6, 6))
    for i, diagram in enumerate(diagrams):
        if len(diagram) > 0:
            birth, death = diagram.T
            plt.scatter(birth, death, color=colors[i % len(colors)], label=f'H{i}')
    plt.plot([0, 1.1], [0, 1.1])  # change according to the dataset
    plt.legend()
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.title(f'Phish persistence diagram for {num} components' if phish else f'Not phish persistence diagram for {num} components')
    plt.ylim(0, 1.1)  # change according to the dataset
    plt.xlim(0, 1.1)  # change according to the dataset
    plt.show()
    def extractH1(diagrams):
        h1Features = diagrams[1]  
        h1_tuples = [(birth, death) for birth, death in h1Features]
        return h1_tuples
    return extractH1(diagrams)


h1P = plots(phishes.head(6000), True, 260)
h1NP = plots(notP.head(6000), False, 260)

