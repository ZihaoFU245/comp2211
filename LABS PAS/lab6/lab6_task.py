

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras import optimizers
from keras import regularizers
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
if __name__ == "__main__":
  # Import libraries
  import matplotlib.pyplot as plt
  import seaborn as sns
  import matplotlib.pyplot as plt

# %%
if __name__ == "__main__":
    # Fetch and download data from sklearn.
    house_dataset = fetch_california_housing(data_home="./", download_if_missing=True)

    # The dataset is already seperated in X and y data
    data = pd.DataFrame(house_dataset.data, columns=house_dataset.feature_names)
    labels = pd.Series(house_dataset.target, name="MedHouseVal")

    # Let us combine the X and y data to have a single dataframe for easier visualization.
    california_data = data.join(labels)

# %%
if __name__ == "__main__":
    print(california_data.head())

# %%
if __name__ == "__main__":
  # Check attribute information
  # As you can see, there are no null values and all are numeric.
  california_data.info()

# %%
if __name__ == "__main__":
  # Visualize distribution for each feature

    california_data.hist(bins=50, figsize=(20,10))
    plt.show()

# %%
if __name__ == "__main__":
  # Visualize correlation with respect to each attribute

  plt.figure(figsize=(10, 8))
  california_data['MedHouseVal'] = labels
  sns.heatmap(california_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
  plt.title('Correlation Heatmap')
  plt.show()

# %% [markdown]
# # Task 1 Data Preprocessing

# %% [markdown]
# The following is what we have learnt from the data exploration above:
# 1. No null values
# 2. All are numeric data types (float64)
# 3. A mix of linear and non-linear relationships is seen in the heatmap.
# 
# According to the feature distributions and the heatmap, we cannot rely on traditional regression models such as Linear Regression due to the presence of non-linear relationships; we would need to use a model which can capture non-linear relationships -> MLP!
# 
# Therefore, the major preprocessing we need to carry out is to normalize our data to standard normal distribution to avoid any bias towards certain features. Additionally, we need to split our data into train and test datasets before building the model.

# %%
def standard_scalar(data : pd.DataFrame, mean : pd.Series, std : pd.Series) -> pd.DataFrame:
    """
    Standardizes the input Pandas DataFrame using Z-score normalization.

    This function normalizes the input data using the provided mean and standard deviation.
    Standardization ensures that each feature has a mean of 0 and a standard deviation of 1,
    which helps improve the performance of machine learning models, especially those relying
    on gradient-based optimization.

    Parameters:
    ----------
    data : pandas.DataFrame
        The dataset containing feature values, where each column represents a feature,
        and each row represents a sample.
    mean : pandas.Series
        The mean values of each feature, computed from the training set.
        Using the training mean ensures consistency and prevents data leakage.
    std : pandas.Series
        The standard deviation of each feature, computed from the training set.
        Avoids data imbalance and ensures a normalized distribution.

    Returns:
    ----------
    standardized_data : pandas.DataFrame
        The standardized dataset with each feature having a mean close to 0
        and a standard deviation close to 1.
    """
    ###############################################################################
    # TODO: your code starts here
    standardized_data = (data - mean) / std

    return standardized_data

# %%
def preprocess(data : pd.DataFrame):

    ###############################################################################
    # TODO: your code starts here
    # Hint: When using train_test_split, please set random_state to 42 to ensure the same dataset split as zinc
    # Hint: Please normalize both the training and test data using the mean and standard deviation of X and y from the training set. Normalizing y helps the model converge. We will grade based on this setting
    X = data.drop(columns=["MedHouseVal"])
    y = data["MedHouseVal"]
    # split data into train and testing sets (use 20% for test size)
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
    # normalize your dataset
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    y_train_mean = y_train.mean(axis=0)
    y_train_std = y_train.std(axis=0)
    # standardize the training data using the mean and std from the training set
    X_train = standard_scalar(X_train, X_train_mean, X_train_std)
    y_train = (y_train - y_train_mean) / y_train_std
    # standardize the test data using the mean and std from the training set
    X_test = standard_scalar(X_test, X_train_mean, X_train_std)
    y_test = (y_test - y_train_mean) / y_train_std
    
    # TODO: your code ends here
    ###############################################################################

    return X_train, X_test, y_train, y_test

# %%
if __name__ == "__main__":
    # Run the function and print the corresponding shapes.
    X_train, X_test, y_train, y_test = preprocess(california_data)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %% [markdown]
# # Task 2 Model Building

# %% [markdown]
# It is now time to finally build our custom MLP model. You can use the following layers to create your model:
# 
# * Fully-connected (`Dense`)
# * Dropout (`Dropout`)
# 
# Additionally, feel free to play around with different activation functions, number of neurons, regularizers, number of hidden layers, etc.
# 
# **Note that you do not have unlimited computing resources and you should avoid creating models that are too large to run on Google Colab.**
# 

# %%
def create_model():
    model = Sequential([
        # Batch size is 32
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001) , input_shape=(8,)),
        layers.Dropout(0.16),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.16),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.16),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        #layers.Dense(2, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(1 , activation='linear', kernel_regularizer=regularizers.l2(0.001))
    ])

    ###############################################################################
    # TODO: your code starts here
    # Hint: you need to take into account the input_shape for the first layer.
    # Hint: Do not create models with more than 10,000 parameters. ZINC will not evaluate models with more than 10,000 parameters.

    # TODO: your code ends here
    ###############################################################################

    return model

# %%
if __name__ == "__main__":
    # Your model summary
    model = create_model()
    model.summary()
    print(type(model))

# %% [markdown]
# # Task 3 Model Compilation
# 

# %% [markdown]
# It is now time to complete the code below to compile your model before training.
# 
# You can use any optimizer, loss function, and metrics to train your model. However, we recommend the following:
# 1. Optimizer = Adam
# 2. Loss = Mean Squared Error
# 3. Metrics = Mean Absolute Error and Mean Squared Error (https://machinelearningmastery.com/regression-metrics-for-machine-learning/)
# 
# Remember to also complete the model.fit function with your hyperparameters. While you are allowed to explore other options, we recommend having the number of epochs close to 100, if not more.

# %%
def model_compile(model : keras.Model):
    ###############################################################################
    # TODO: your code starts here
    
    # Fill the model.compile() function below
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mean_absolute_error", "mean_squared_error"]
    )
    # TODO: your code ends here
    ###############################################################################
    return model

# %%
if __name__ == "__main__":
  model = model_compile(model)

# %%
if __name__ == "__main__":
  # You can adjust the epochs, batch size, and validation_split to achieve better results.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    history = model.fit(
        X_train, y_train, 
        epochs=100,  # Increased epochs
        batch_size=64,  # Smaller batch size for better generalization
        validation_split=0.2,
        callbacks=[reduce_lr]
    )
    model.save_weights("mlp_model.weights.h5")

# %% [markdown]
# # Model Evaluation

# %% [markdown]
# It is now time to run the code cells below to generate predictions of house prices. This will then be evaluated with the testing labels using 3 metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R2 score.
# 
# The R2 score is a statistical measure that tells us how well our model fits the data. It has a range between 0 and 1, with 1 indicating that our model fits perfectly well with the data. It is important to note that a negative R2 value means your model has not understood the data distribution at all.
# 
# Finally, you can run the last code cell to plot an Actual vs. Predicted values graph. This data visualization technique is very useful in regression tasks, as it showcases how well the predictions fit the regressed diagonal line. If your prediction points are close to the diagonal line, it means you have a high R2 score.
# 
# **Do not change the code cells in this section.**

# %%
if __name__ == "__main__":
  # Predict median house prices based on testing data
  # model.load_weights("mlp_model.weights.h5")
  y_pred = model.predict(X_test).flatten()

  # Generate MSE, RMSE, and R2 scores.
  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  r2 = r2_score(y_test, y_pred)
  # The r2 results should be the same locally and on Zinc, as long as the code is complete and follows the requirements.
  print("mse =", mse, ", rmse =", rmse, ", r2 =", r2)
  # 0.7942

# %%
if __name__ == "__main__":
  # Plot Actual Vs. Predicted Median House Values
  plt.figure(figsize=(8, 6))
  plt.scatter(y_test, y_pred, alpha=0.5)
  plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Diagonal line
  plt.title('Actual vs Predicted Median House Value')
  plt.xlabel('Actual Values')
  plt.ylabel('Predicted Values')
  plt.show()

# %% [markdown]
# ## **Grading Scheme**
# 
# Please export your notebook on Colab as `lab6_tasks.py` (File -> Download -> Download .py), and submit it together with your `mlp_model.weights.h5` model weight file.
# 
# 
# * You get **3 points** for data preprocessing (task 1)
# * You get **2 points** for the valid implementation of the MLP model (task 2)
# * You get **1 point** for model compilation (task 3)
# * You get **1 point** for achieving an R2 score of at least 0.70
# * You get **2 points** for achieving an R2 score of at least 0.75
# * You get **3 points** (full mark) for achieving an R2 score of at least 0.78


