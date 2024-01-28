import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_digits(return_X_y=True)

# Count the number of instances of each digit
digit_counts = [np.sum(y == i) for i in range(10)]
digits = np.arange(10)

# Plot the distribution of labels
plt.figure(figsize=(8, 4))
plt.bar(digits, digit_counts, color='skyblue')
plt.title('Distribution of Digits')
plt.xlabel('Digit')
plt.ylabel('Number of Instances')
plt.xticks(digits)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Group images by their corresponding labels
vec_img = {i: X_train[y_train==i].T for i in range(10)} 

# Classify an input image using Singular Value Decomposition (SVD)
def SVD_classifier(img_classes, z, k=10):
    # Initialize minimal residual and label
    min_res = np.infty
    label = -1
    
    # Iterate over each class
    for i in range(10):
        # Perform SVD on images of the class
        U, S, Vt = np.linalg.svd(img_classes[i])
        
        # Calculate the residual
        res = np.linalg.norm((np.eye(U.shape[0]) - U[:, :k] @ U[:, :k].T) @ z)
        
        # Update minimal residual and label if necessary
        if res < min_res:
            min_res = res
            label = i
            
    return label

# Calculate accuracy of predictions
def get_accuracy(y_test, y_pred):
    return 100 * np.sum(np.asarray(y_pred) == y_test) / y_test.shape[0]

# Test different values of k for SVD
k_list = []
acc_list = []
for k in range(2, 31, 2):
    k_list.append(k)
    
    # Make predictions
    y_pred = [SVD_classifier(vec_img, X_test[i, :], k=k) for i in range(y_test.shape[0])]
    
    # Calculate accuracy
    accuracy = get_accuracy(y_test, y_pred)
    acc_list.append(accuracy)
    print(f"K: {k}, Accuracy: {accuracy:.2f}%")

# Plot accuracy vs number of components (k) in SVD
plt.plot(k_list, acc_list, c='skyblue', linewidth=2, marker='o', markersize=6, label='Accuracy')
plt.title('Accuracy vs Number of Components (K) in SVD')
plt.xlabel('Number of Components (K)')
plt.ylabel('Accuracy (%)')
plt.grid(linestyle='--', alpha=0.7)
plt.xticks(np.arange(0, 51, 5))  # Adjust x-axis ticks for better readability

# Highlight maximum accuracy point
max_accuracy = max(acc_list)
max_index = acc_list.index(max_accuracy)
plt.scatter(k_list[max_index], max_accuracy, color='red', zorder=5, label=f'Max Accuracy ({max_accuracy:.2f}%)')

plt.legend()
plt.tight_layout()
plt.show()