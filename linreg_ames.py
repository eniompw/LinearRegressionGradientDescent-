from sklearn.datasets import fetch_openml          # dataset loader from scikit-learn (Ames housing prices)
import numpy as np                                 # standard NumPy numerical computing library

# --- Data ---
ames = fetch_openml(name="house_prices", as_frame=True)  # fetch Ames housing dataset from OpenML as a DataFrame
X = ames.data["GrLivArea"].to_numpy(dtype=float)  # extract above-ground living area feature (sq ft)
y = ames.target.to_numpy(dtype=float)             # extract sale price target values (USD)

# --- Normalise ---
X_norm = (X - X.mean()) / X.std()                 # standardize input feature (mean=0, std=1)
y_norm = (y - y.mean()) / y.std()                 # standardize target values (mean=0, std=1)

# --- Model ---
W, b = 0.0, 0.0                                   # initialise weight and bias to zero
lr, n = 0.1, len(X)                               # learning rate and number of training samples

# --- Train ---
for _ in range(1000):                             # train for 1000 epochs (full batch gradient descent)
    # Forward pass
    y_pred = W * X_norm + b                       # compute predicted normalised sale price

    # Backward pass
    error = y_norm - y_pred                       # residual between actual and predicted values
    W -= lr * -(2/n) * np.sum(X_norm * error)     # gradient descent step on weight parameter
    b -= lr * -(2/n) * np.sum(error)              # gradient descent step on bias term

# --- Denormalise ---
slope     = W * y.std() / X.std()                 # rescale weight back to original feature/target units
intercept = b * y.std() + y.mean() - slope * X.mean()  # recover intercept in original price scale

print(f"slope:     {slope:.4f}")                  # price increase per additional sq ft of living area
print(f"intercept: {intercept:.4f}")              # predicted price when living area is zero
