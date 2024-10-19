import numpy as np
import os

file_path = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs/Vectorized/Training/PD_slices/'

# Try loading the file
flag = True
start = 0
while flag:
    try:
        for i in range(start, 40993):
            file = file_path + f'{i}.npz'
            if not os.path.exists(file):
                continue
            data = np.load(file, allow_pickle=True)
            if i == 40992:
                print('finito')
                flag = False
            #print(f"Keys in the file: {data.files}")  # Prints the names of arrays in the .npz file
            #for key in data.files:
                #print(f"Array '{key}': {data[key].shape}")  # Prints the shape of each array
    except Exception as e:
        os.remove(file)
        start = i+1
        print(f"Removing file: {i}")

print('todo bien')

