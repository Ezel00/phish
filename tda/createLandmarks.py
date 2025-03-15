import pandas as pd
import numpy as np
import os


path1=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet2\dfPCA12Train.csv"
path2 = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet2\dfPCA12Train.csv"
dir1 = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\embed\PCA12\phish"
dir2 = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\embed\PCA12\notPhish"

#extracts the phish or not phish dataframes from the larger df
# input path and 1 or 0, 1 gets phish
def preProcess(path, phish):
    df = pd.read_csv(path)
    phish = df[df['label'] == phish]
    phish = phish.drop(columns=['label'])
    return phish.to_numpy()

phish = preProcess(path1, 1)
notP = preProcess(path1, 0)

#needs to have the data as numpy arrays, without the labels
def saveLand(dirPhish, dirNotPhish, phishes, notPhishes):
    # Create directories if they don’t exist
    os.makedirs(dirPhish, exist_ok=True)
    os.makedirs(dirNotPhish, exist_ok=True)
    # input allPts, #toselect
    def pick_n_random_points(points, nb_points):
        indices = np.random.choice(len(points), size=min(nb_points, len(points)), replace=False)
        return points[indices]
    for i in range(500):
        # Select landmarks
        landmarksPhish = pick_n_random_points(phishes, nb_points=1500)
        landmarksNotP = pick_n_random_points(notPhishes, nb_points=1500)
        # Save landmarks to CSV
        np.savetxt(fr"{dirPhish}\Phish_{i+1}.csv", landmarksPhish, delimiter=",")
        np.savetxt(fr"{dirNotPhish}\NotP_{i+1}.csv", landmarksNotP, delimiter=",")
        print(f"{i} done")
    print("Landmark selection completed.")

saveLand(dir1, dir2, phish, notP)
