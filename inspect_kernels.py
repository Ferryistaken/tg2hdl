"""Inspect tinygrad kernels for a minimal MNIST model.

Builds a simple MNIST-style network and dumps all kernel summaries.
"""
import os
os.environ['DEBUG'] = '2'  # show kernel summaries without full source

from tinygrad import Tensor

print("=" * 60)
print("BUILDING MODEL")
print("=" * 60)

# Use fixed data instead of randn to avoid PRNG kernels cluttering output
# In real inference these would be loaded from memory, not generated
Tensor.manual_seed(42)

# Input: batch of 1 image, 28x28 flattened
x = Tensor.randn(1, 784)

# Layer 1: Linear(784, 128) + ReLU
w1 = Tensor.randn(784, 128)
b1 = Tensor.zeros(128)

# Layer 2: Linear(128, 10)
w2 = Tensor.randn(128, 10)
b2 = Tensor.zeros(10)

print("\n" + "=" * 60)
print("REALIZING WEIGHTS (PRNG kernels - ignore these)")
print("=" * 60)
x.realize(), w1.realize(), b1.realize(), w2.realize(), b2.realize()

print("\n" + "=" * 60)
print("FORWARD PASS KERNELS (these are what we care about)")
print("=" * 60)

h = (x @ w1 + b1).relu()
logits = h @ w2 + b2
output = logits.log_softmax()
output.realize()

# Now let's also look at the schedule to understand the ops
print("\n" + "=" * 60)
print("FORWARD PASS SCHEDULE DETAILS")
print("=" * 60)

# Rebuild to get schedule
h2 = (x @ w1 + b1).relu()
logits2 = h2 @ w2 + b2
output2 = logits2.log_softmax()

sched = output2.schedule()
for i, si in enumerate(sched):
    print(f"\nKernel {i}: {si}")
