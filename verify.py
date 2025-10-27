import torch

file_path = r'C:\Users\kande\Desktop\Projects\DeepfakeDetector\best_model.pth'
try:
    # Attempt to load the model state dictionary
    state_dict = torch.load(file_path)
    print("File loaded successfully! It appears to be a valid PyTorch model state dictionary.")
    # You can optionally print some keys to inspect the content
    # print("Keys in the state dictionary:", state_dict.keys())
except Exception as e:
    print(f"Error loading the file: {e}")
    print("The file might be corrupted or not a valid PyTorch model state dictionary.")
