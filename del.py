import os

# Define the folder path containing the files
folder_path = '/home/httpiego/PycharmProjects/VulDeeDiegator/iSeVCs/Vectorized/Training/AD_slices'

files = os.listdir(folder_path)

# Loop through each file and check if its size is 0 bytes
for filename in files:
    file_path = os.path.join(folder_path, filename)
    # Check if it's a file and if its size is 0
    if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
        # Delete the file
        os.remove(file_path)
        print(f"Deleted: {filename}")

print("Deletion of zero-size files complete.")