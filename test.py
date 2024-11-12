import os
import random

train_folder = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs/Vectorized/Training/'

# Get the list of files in the folder
file_list = os.listdir(train_folder)

# Shuffle the list in place
random.shuffle(file_list)

# Now 'file_list' is shuffled, print it
print(file_list)
