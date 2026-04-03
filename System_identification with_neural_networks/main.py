'''Building a neural network that learns the physics of a mass-spring damper
   purely from data. Once trained, the NN acts as a "learned model" that predicts
   how the system evolves over time.

   - Data collection from mass-spring-damper ODE simulation with random intial
     conditions and force.

   - Normalizing and spliting data into train / val sets for PyTorch integration
   
   - Bulidinf a NN architecture with 3 inputs -> 64 -> 64 -> 2 outputs.
   
   - Training the NN model with obtained data.
   
   - Evaluation with different initial conditions and force
'''
import numpy as np
import subprocess
import os
import sys
# for calling and saving the files to the path
BASE = os.path.dirname(os.path.abspath(__file__))

# Active Python env call
PYTHON = sys.executable

print("=== Step 1: Generating simulation data ===")
from simulate import generate_data
data = generate_data(n_sequences=200)
np.save(os.path.join(BASE, "system_data.npy"), data)
print(f"Saved {len(data)} samples\n")

print("=== Step 2-4: Training neural network ===")
subprocess.run([PYTHON, os.path.join(BASE, "train.py")], cwd=BASE)

print("\n=== Step 5: Evaluating rollout ===")
subprocess.run([PYTHON, os.path.join(BASE, "evaluate.py")], cwd=BASE)

print("\nDone!")