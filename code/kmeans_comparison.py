"""
K-Means Clustering Comparison
Compare unsupervised k-means with supervised neural network on Iris dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from scipy.stats import mode
import os

# set random seed
np.random.seed(42)

# create plots directory
os.makedirs('../plots', exist_ok=True)

# load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Total samples: {len(X)}")
print(f"Features: {iris.feature_names}")
print(f"Classes: {list(class_names)}")

# Part 1: K-Means Clustering (Unsupervised)
print("\n" + "="*50)
print("K-MEANS CLUSTERING (UNSUPERVISED)")
print("="*50)

# fit k-means with k=3 (we know there are 3 species)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# map clusters to actual labels using majority vote
# for each cluster, find which true label is most common
def map_clusters_to_labels(cluster_labels, true_labels):
    """Map cluster assignments to true labels by majority vote"""
    mapped_labels = np.zeros_like(cluster_labels)

    for cluster_id in range(3):
        # find all samples in this cluster
        mask = cluster_labels == cluster_id
        # find the most common true label in this cluster
        if np.sum(mask) > 0:
            true_labels_in_cluster = true_labels[mask]
            # use mode to find most common label
            most_common = mode(true_labels_in_cluster, keepdims=False)
            mapped_labels[mask] = most_common.mode

    return mapped_labels

# get mapped predictions
kmeans_predictions = map_clusters_to_labels(cluster_labels, y)
kmeans_accuracy = accuracy_score(y, kmeans_predictions)

print(f"K-Means Accuracy (with majority vote mapping): {kmeans_accuracy:.4f}")

# show cluster to label mapping
for cluster_id in range(3):
    mask = cluster_labels == cluster_id
    if np.sum(mask) > 0:
        true_labels_in_cluster = y[mask]
        most_common = mode(true_labels_in_cluster, keepdims=False)
        print(f"  Cluster {cluster_id} → {class_names[most_common.mode]} (majority)")

# Part 2: Neural Network (Supervised)

# prepare data for neural network
y_onehot = to_categorical(y, num_classes=3)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_onehot, test_size=0.2, random_state=42
)
y_test_labels = np.argmax(y_test, axis=1)

# build a simple neural network (same as our experiments)
model = Sequential([
    Dense(8, input_shape=(4,), activation='relu'),
    Dense(3, activation='softmax')
])

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
    verbose=0
)

# evaluate on test set
test_loss, nn_test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Neural Network Test Accuracy: {nn_test_accuracy:.4f}")

# also evaluate on full dataset for fair comparison with k-means
y_full_onehot = to_categorical(y, num_classes=3)
_, nn_full_accuracy = model.evaluate(X_scaled, y_full_onehot, verbose=0)
print(f"Neural Network Full Dataset Accuracy: {nn_full_accuracy:.4f}")

# Part 3: Comparison
print("COMPARISON: K-MEANS vs NEURAL NETWORK")

print(f"\nK-Means Accuracy (full dataset):     {kmeans_accuracy:.4f} ({kmeans_accuracy*100:.1f}%)")
print(f"Neural Network Accuracy (test set):  {nn_test_accuracy:.4f} ({nn_test_accuracy*100:.1f}%)")
print(f"Neural Network Accuracy (full data): {nn_full_accuracy:.4f} ({nn_full_accuracy*100:.1f}%)")

improvement = nn_full_accuracy - kmeans_accuracy
print(f"\nNeural Network improvement over K-Means: {improvement:.4f} ({improvement*100:.1f}%)")

# Plots

# Plot 1: Bar chart comparison
fig, ax = plt.subplots(figsize=(8, 6))

methods = ['K-Means\n(Unsupervised)', 'Neural Network\n(Supervised)']
accuracies = [kmeans_accuracy, nn_full_accuracy]
colors = ['coral', 'steelblue']

bars = ax.bar(methods, accuracies, color=colors, width=0.5, edgecolor='black')

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('K-Means vs Neural Network on Iris Dataset', fontsize=14)
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

# add value labels
for bar, acc in zip(bars, accuracies):
    ax.annotate(f'{acc:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, acc),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../plots/kmeans_vs_nn_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Scatter plot showing clustering vs true labels
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# use first two features for visualization (sepal length and sepal width)
feature_x, feature_y = 0, 1  # sepal length, sepal width

# left plot: K-means clusters
ax1 = axes[0]
scatter1 = ax1.scatter(X_scaled[:, feature_x], X_scaled[:, feature_y],
                       c=cluster_labels, cmap='viridis', alpha=0.7, s=50)
ax1.scatter(kmeans.cluster_centers_[:, feature_x], kmeans.cluster_centers_[:, feature_y],
            c='red', marker='X', s=200, edgecolors='black', linewidths=2, label='Centroids')
ax1.set_xlabel('Sepal Length (normalized)')
ax1.set_ylabel('Sepal Width (normalized)')
ax1.set_title('K-Means Clustering')
ax1.legend()

# right plot: True labels
ax2 = axes[1]
scatter2 = ax2.scatter(X_scaled[:, feature_x], X_scaled[:, feature_y],
                       c=y, cmap='viridis', alpha=0.7, s=50)
ax2.set_xlabel('Sepal Length (normalized)')
ax2.set_ylabel('Sepal Width (normalized)')
ax2.set_title('True Labels')

# add colorbar
cbar = plt.colorbar(scatter2, ax=ax2, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(class_names)

plt.tight_layout()
plt.savefig('../plots/kmeans_clusters_visualization.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Confusion matrix for k-means
fig, ax = plt.subplots(figsize=(6, 5))

cm = confusion_matrix(y, kmeans_predictions)
im = ax.imshow(cm, cmap='Blues')

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('K-Means Confusion Matrix')

# add text annotations
for i in range(3):
    for j in range(3):
        text = ax.text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)

plt.colorbar(im)
plt.tight_layout()
plt.savefig('../plots/kmeans_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Save results to text file
with open('../plots/kmeans_results.txt', 'w') as f:
    f.write("K-MEANS vs NEURAL NETWORK COMPARISON\n")
    f.write("="*50 + "\n\n")
    f.write(f"K-Means Accuracy (unsupervised): {kmeans_accuracy:.4f}\n")
    f.write(f"Neural Network Accuracy (supervised): {nn_full_accuracy:.4f}\n")
    f.write(f"Improvement with supervision: {improvement:.4f}\n\n")
    f.write("Note: K-Means clusters were mapped to true labels using majority vote.\n")
    f.write("This represents the best possible accuracy for k-means on this task.\n")