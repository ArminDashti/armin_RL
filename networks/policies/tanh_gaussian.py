import torch

# Set the number of threads
# torch.set_num_threads(4)

# Get the current number of threads
num_threads = torch.get_num_threads()
print(f"Number of threads: {num_threads}")
