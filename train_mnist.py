"""Train a 2-layer MLP on MNIST and save weights."""

from tinygrad import Tensor, nn
from tinygrad.nn.datasets import mnist
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_state_dict, safe_save

class MNISTNet:
    def __init__(self):
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)

    def __call__(self, x):
        return self.l2(self.l1(x).relu())

def train():
    model = MNISTNet()
    opt = SGD(nn.state.get_parameters(model), lr=0.01, momentum=0.9)

    X_train, y_train, X_test, y_test = mnist()

    # Normalize to [0, 1] float32
    X_train = X_train.reshape(-1, 784).cast("float32") / 255.0
    X_test = X_test.reshape(-1, 784).cast("float32") / 255.0

    EPOCHS = 5
    BS = 128

    for epoch in range(EPOCHS):
        Tensor.training = True
        total_loss = 0.0
        n_batches = 0

        for i in range(0, X_train.shape[0], BS):
            xb = X_train[i:i+BS]
            yb = y_train[i:i+BS]

            logits = model(xb)
            loss = logits.sparse_categorical_crossentropy(yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        # Test accuracy
        Tensor.training = False
        test_logits = model(X_test)
        test_pred = test_logits.argmax(axis=1)
        acc = (test_pred == y_test).mean().item()

        print(f"Epoch {epoch+1}/{EPOCHS}  loss={total_loss/n_batches:.4f}  test_acc={acc:.4f}")

    # Save weights
    state = get_state_dict(model)
    safe_save(state, "mnist_weights.safetensors")
    print(f"\nSaved weights to mnist_weights.safetensors")
    print(f"Keys: {list(state.keys())}")
    for k, v in state.items():
        print(f"  {k}: shape={v.shape}")

if __name__ == "__main__":
    train()
