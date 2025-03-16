import numpy as np

def dirtyY(yT, flip, random_seed=42):
    np.random.seed(random_seed)
    yT = np.array(yT)
    zeros = np.where(yT == 0)[0]
    ones = np.where(yT == 1)[0]
    num_zeros = int(len(zeros) * flip)
    num_ones = int(len(ones) * flip)
    zero_inds = np.random.choice(zeros, num_zeros, replace=False)
    one_inds = np.random.choice(ones, num_ones, replace=False)
    yT[zero_inds] = 1  # Flip 0 -> 1
    #yT[one_inds] = 0  # Flip 1 -> 0
    print(f"Original Label Distribution: 0s = {len(zeros)}, 1s = {len(ones)}")
    new_zeros = np.sum(yT == 0)
    new_ones = np.sum(yT == 1)
    print(f"Dirty Label Distribution: 0s = {new_zeros}, 1s = {new_ones}")
    return yT



