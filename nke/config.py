"""Global configuration and paths for the NKE experiment suite."""
import os
import torch

# --- paths ---
ROOT = "/home/akayyam/adversarial"
# CIFAR-10 / MNIST were downloaded into the session scratchpad; reuse them.
DATA_DIR = "/tmp/claude-1063/-home-akayyam-adversarial/4dcd285a-a3b7-48c6-87c8-daa791c4046c/scratchpad/data"
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
STIM_DIR = os.path.join(ROOT, "stimuli")

for _d in (CKPT_DIR, RESULTS_DIR, FIG_DIR, STIM_DIR):
    os.makedirs(_d, exist_ok=True)

# --- device: GPU 3 is the free card on this box ---
if torch.cuda.is_available():
    # pick the GPU with the most free memory
    free = []
    for i in range(torch.cuda.device_count()):
        try:
            f, _ = torch.cuda.mem_get_info(i)
        except Exception:
            f = 0
        free.append(f)
    DEVICE = torch.device(f"cuda:{int(max(range(len(free)), key=lambda i: free[i]))}")
else:
    DEVICE = torch.device("cpu")

SEED = 1234

# dataset normalization stats
NORM = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}
NUM_CLASSES = {"mnist": 10, "cifar10": 10}
IN_CHANS = {"mnist": 1, "cifar10": 3}
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
MNIST_CLASSES = [str(i) for i in range(10)]
CLASS_NAMES = {"mnist": MNIST_CLASSES, "cifar10": CIFAR10_CLASSES}
