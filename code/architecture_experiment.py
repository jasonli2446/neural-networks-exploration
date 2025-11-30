"""
Network Architecture Experiment
Compare shallow, wide, and deep networks on the Iris dataset
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

# normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# one-hot encode labels
y_onehot = to_categorical(y, num_classes=3)

# split data: 60% train, 20% val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# define different architectures
architectures = {
    'shallow': {
        'name': 'Shallow (4→4→3)',
        'layers': [4]  # just one hidden layer with 4 neurons
    },
    'wide': {
        'name': 'Wide (4→16→3)',
        'layers': [16]  # one hidden layer with 16 neurons
    },
    'deep': {
        'name': 'Deep (4→8→8→3)',
        'layers': [8, 8]  # two hidden layers with 8 neurons each
    }
}

# store results
results = {}
histories = {}

# train each architecture
for arch_key, arch_config in architectures.items():
    print(f"Training {arch_config['name']}")

    # build the model
    model = Sequential()

    # add first hidden layer with input shape
    model.add(Dense(arch_config['layers'][0], input_shape=(4,), activation='relu'))

    # add additional hidden layers if any
    for units in arch_config['layers'][1:]:
        model.add(Dense(units, activation='relu'))

    # add output layer
    model.add(Dense(3, activation='softmax'))

    # print model summary
    print(f"Architecture: 4 → {' → '.join(map(str, arch_config['layers']))} → 3")

    # compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # train
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
    results[arch_key] = {
        'name': arch_config['name'],
        'train_acc': history.history['accuracy'][-1],
        'val_acc': history.history['val_accuracy'][-1],
        'test_acc': test_acc,
        'train_loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1]
    }
    histories[arch_key] = history.history

    print(f"Final Training Accuracy: {results[arch_key]['train_acc']:.4f}")
    print(f"Final Validation Accuracy: {results[arch_key]['val_acc']:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

# Plot 1: Learning curves for each architecture
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (arch_key, arch_config) in enumerate(architectures.items()):
    ax = axes[idx]
    ax.plot(histories[arch_key]['accuracy'], label='Training', linewidth=2)
    ax.plot(histories[arch_key]['val_accuracy'], label='Validation', linewidth=2)
    ax.set_title(arch_config['name'], fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('../plots/architecture_learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Loss curves to show overfitting
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (arch_key, arch_config) in enumerate(architectures.items()):
    ax = axes[idx]
    ax.plot(histories[arch_key]['loss'], label='Training Loss', linewidth=2)
    ax.plot(histories[arch_key]['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_title(arch_config['name'], fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/architecture_loss_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Bar chart comparing final accuracies
fig, ax = plt.subplots(figsize=(10, 6))

arch_names = [results[k]['name'] for k in architectures.keys()]
x = np.arange(len(arch_names))
width = 0.25

train_accs = [results[k]['train_acc'] for k in architectures.keys()]
val_accs = [results[k]['val_acc'] for k in architectures.keys()]
test_accs = [results[k]['test_acc'] for k in architectures.keys()]

bars1 = ax.bar(x - width, train_accs, width, label='Training', color='steelblue')
bars2 = ax.bar(x, val_accs, width, label='Validation', color='darkorange')
bars3 = ax.bar(x + width, test_accs, width, label='Test', color='forestgreen')

ax.set_ylabel('Accuracy')
ax.set_title('Architecture Comparison - Final Accuracies')
ax.set_xticks(x)
ax.set_xticklabels(arch_names)
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

# add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../plots/architecture_comparison_bars.png', dpi=150, bbox_inches='tight')
plt.close()

# Print summary table
print("ARCHITECTURE EXPERIMENT - SUMMARY")
print(f"{'Architecture':<20} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
for arch_key in architectures.keys():
    print(f"{results[arch_key]['name']:<20} {results[arch_key]['train_acc']:<12.4f} {results[arch_key]['val_acc']:<12.4f} {results[arch_key]['test_acc']:<12.4f}")

# Check for overfitting (train acc much higher than val acc)
print("\nOverfitting Analysis:")
for arch_key in architectures.keys():
    gap = results[arch_key]['train_acc'] - results[arch_key]['val_acc']
    status = "Possible overfitting" if gap > 0.05 else "Good generalization"
    print(f"  {results[arch_key]['name']}: Train-Val gap = {gap:.4f} ({status})")

# Save results to text file
with open('../plots/architecture_results.txt', 'w') as f:
    f.write("ARCHITECTURE EXPERIMENT RESULTS\n")
    f.write(f"{'Architecture':<20} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}\n")
    for arch_key in architectures.keys():
        f.write(f"{results[arch_key]['name']:<20} {results[arch_key]['train_acc']:<12.4f} {results[arch_key]['val_acc']:<12.4f} {results[arch_key]['test_acc']:<12.4f}\n")