
import pandas as pd
import numpy as np
import os
import random
from ripser import ripser

path1=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet2\dfPCA12Train.csv"
path2 = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet2\dfPCA12Train.csv"

inpDir = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\embed\PCA12\phish"
dir2 = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\embed\PCA12\phish"

def stats(inpDir, outDir, name):
    def Ripser_Code():
        for i in range(500):  # Loop from Phish_1.csv to Phish_500.csv
            filename = f'{inpDir}\{name}_{i + 1}.csv'
            try:
                data = pd.read_csv(filename, header=None)
                print(f"Processing file: {filename}, Shape: {data.shape}")

                data = data.dropna()  # Ensure no missing values
                if data.empty:
                    print(f"File {filename} is empty after dropping NaN values!")
                    continue

                data = data.dropna(axis=1, how="all")  # Drop fully empty columns
                if data.shape[1] == 0:
                    print(f"File {filename} has no valid features after dropping NaN columns!")
                    continue

                landmark_size = len(data)
                chunk_size = max(1, landmark_size // 20)  # Ensure at least one row per chunk
                row_start = 0
                row_stop = chunk_size

                for j in range(20):
                    landmark_set = data.iloc[row_start:row_stop, :]
                    print(
                        f"Iteration {j}: row_start={row_start}, row_stop={row_stop}, landmark_set.shape={landmark_set.shape}")

                    if landmark_set.empty:
                        print(f"Landmark set is empty for {filename} in iteration {j}")
                        break

                    points = ripser(landmark_set)['dgms']

                    if len(points) < 2 or len(points[0]) == 0 or len(points[1]) == 0:
                        print(f"Ripser failed to compute persistence for {filename} in iteration {j}")
                        continue

                    points_h0 = np.array(points[0])
                    points_h1 = np.array(points[1])

                    points_h0_0 = points_h0[:, 0]
                    points_h0_1 = points_h0[:, 1]
                    points_h1_0 = points_h1[:, 0]
                    points_h1_1 = points_h1[:, 1]

                    points_h0_1[~np.isfinite(points_h0_1)] = 1.41421
                    points_h1_0[~np.isfinite(points_h1_0)] = 1.41421
                    points_h1_1[~np.isfinite(points_h1_1)] = 1.41421

                    length_0 = abs(points_h0_1 - points_h0_0)
                    y_max_0 = np.max(points_h0_1)
                    ymlength_0 = y_max_0 - points_h0_1

                    length_1 = abs(points_h1_1 - points_h1_0)
                    y_max_1 = np.max(points_h1_1)
                    ymlength_1 = y_max_1 - points_h1_1

                    M0 = [np.mean(points_h0_0), np.mean(points_h0_1), np.mean(length_0), np.mean(ymlength_0),
                          np.median(points_h0_0), np.median(points_h0_1), np.median(length_0), np.median(ymlength_0),
                          np.std(points_h0_0), np.std(points_h0_1), np.std(length_0), np.std(ymlength_0)]

                    M1 = [np.mean(points_h1_0), np.mean(points_h1_1), np.mean(length_1), np.mean(ymlength_1),
                          np.median(points_h1_0), np.median(points_h1_1), np.median(length_1), np.median(ymlength_1),
                          np.std(points_h1_0), np.std(points_h1_1), np.std(length_1), np.std(ymlength_1)]

                    M = np.concatenate((M0, M1), axis=None)
                    M = np.append(M, i + 1)

                    Output = ','.join(['%.7f' % num for num in M])
                    output_filename = f'{outDir}\Ripser_20iter_50perc.txt'
                    with open(output_filename, 'a') as file:
                        file.write('%s\n' % Output)

                    row_start = row_stop
                    row_stop = min(row_stop + chunk_size, len(data))  # Avoid out-of-bounds indexing

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue

        filename = f'{outDir}\Ripser_20iter_50perc.txt'
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            random.shuffle(lines)
            randomized_filename = f'{outDir}\Ripser_20iter_50perc_random.txt'
            with open(randomized_filename, 'w') as newfile:
                newfile.writelines(lines)
        except Exception as e:
            print(f"Error randomizing file {filename}: {e}")

    Ripser_Code()

#change Phish to NotP when needed
stats(inpDir,inpDir, 'Phish')


