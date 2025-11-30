"""
Activation Function Experiment
Compare sigmoid, tanh, and ReLU on the Iris dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import os

# set random seed for reproducibility
np.random.seed(42)

# create plots directory if it doesn't exist
os.makedirs('../plots', exist_ok=True)

# load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# convert labels to one-hot encoding for neural network
y_onehot = to_categorical(y, num_classes=3)

# split into train, validation, and test sets
# first split: 80% train+val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)
# second split: 75% train, 25% val (of the 80%, so 60% train, 20% val overall)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# define the activation functions to test
activations = ['sigmoid', 'tanh', 'relu']

# store results
results = {}
histories = {}

# train a network for each activation function
for activation in activations:
    print(f"Training with {activation} activation")

    # build the model. same architecture for all (4 -> 8 -> 3)
    model = Sequential([
        Dense(8, input_shape=(4,), activation=activation),
        Dense(3, activation='softmax')
    ])

    # compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=8,
        validation_data=(X_val, y_val),
        verbose=0
    )

    # evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # store results
    results[activation] = {
        'train_acc': history.history['accuracy'][-1],
        'val_acc': history.history['val_accuracy'][-1],
        'test_acc': test_acc
    }
    histories[activation] = history.history

    print(f"Final Training Accuracy: {results[activation]['train_acc']:.4f}")
    print(f"Final Validation Accuracy: {results[activation]['val_acc']:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

# Plot 1: Learning curves for each activation function
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, activation in enumerate(activations):
    ax = axes[idx]
    ax.plot(histories[activation]['accuracy'], label='Training', linewidth=2)
    ax.plot(histories[activation]['val_accuracy'], label='Validation', linewidth=2)
    ax.set_title(f'{activation.upper()} Activation', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('../plots/activation_learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Bar chart comparing final accuracies
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(activations))
width = 0.25

train_accs = [results[a]['train_acc'] for a in activations]
val_accs = [results[a]['val_acc'] for a in activations]
test_accs = [results[a]['test_acc'] for a in activations]

bars1 = ax.bar(x - width, train_accs, width, label='Training', color='steelblue')
bars2 = ax.bar(x, val_accs, width, label='Validation', color='darkorange')
bars3 = ax.bar(x + width, test_accs, width, label='Test', color='forestgreen')

ax.set_ylabel('Accuracy')
ax.set_title('Activation Function Comparison - Final Accuracies')
ax.set_xticks(x)
ax.set_xticklabels([a.upper() for a in activations])
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

# add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../plots/activation_comparison_bars.png', dpi=150, bbox_inches='tight')
plt.close()

# Print summary table
print("ACTIVATION FUNCTION EXPERIMENT - SUMMARY")
print(f"{'Activation':<12} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
for activation in activations:
    print(f"{activation.upper():<12} {results[activation]['train_acc']:<12.4f} {results[activation]['val_acc']:<12.4f} {results[activation]['test_acc']:<12.4f}")

# Save results to a text file for easy reference
with open('../plots/activation_results.txt', 'w') as f:
    f.write("ACTIVATION FUNCTION EXPERIMENT RESULTS\n")
    f.write(f"{'Activation':<12} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}\n")
    for activation in activations:
        f.write(f"{activation.upper():<12} {results[activation]['train_acc']:<12.4f} {results[activation]['val_acc']:<12.4f} {results[activation]['test_acc']:<12.4f}\n")