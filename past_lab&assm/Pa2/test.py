import numpy as np

X = np.array(
    [[0,0],
     [0.1,0.1],
     [0.2,0.2]]
)

y = np.array(
    [[0,0.1],
     [0.2,0.3]]
)

# Compute the denominator safely to avoid invalid operations
denominator = 1 - np.sum(X**2, axis=1)[:, None] * (1 - np.sum(y**2, axis=1)[None, :])
denominator = np.where(denominator <= 0, np.nan, denominator)  # Avoid division by zero or invalid values

# Compute the distance using the corrected denominator
d = np.acosh(1 + 2 * (np.sum((X[:, None, :] - y[None, :, :]) ** 2, axis=2) / denominator))

print(d)


