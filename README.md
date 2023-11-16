# DeepLearning-Project


Required steps to test the model

run the following code from collab
https://colab.research.google.com/drive/1zPdL7nuIF45Gq28PNilD9X9z-VGD6NTF?usp=sharing


Upload models for any step size and city per your need in Google Colab. I have saved every model named model as B_1. Here, B and 1 in model B_1 indicate that the model is trained on the city B dataset, considering step size 1. So select any model with step sizes 1,7,14,30,60 or with different cities B,G,S,T from the file that I have provided.


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.optimizers import Adam
import matplotlib.pyplot as plt

model = load_model('Cityname_stepsize.h5')

# Load your test dataset for City B (replace 'B_test.csv' with the actual file name)
df_test = pd.read_csv('your_test_file')
df_test.drop('Unnamed: 0', axis=1, inplace=True)
X_test = df_test.to_numpy()
y_test = df_test['PM25_Concentration'].to_numpy()
def get_processed_test_data(X_test, y_test, sequence_length, step_size):
    X_test_processed = []
    y_test_processed = []

    for i in range(0, X_test.shape[0] - sequence_length - step_size):
        X_test_processed.append(X_test[i : i + sequence_length])
        y_test_processed.append(y_test[i + sequence_length: i + sequence_length + step_size].tolist())

    X_test_processed = np.array(X_test_processed)
    y_test_processed = np.array(y_test_processed)

    return X_test_processed, y_test_processed

# +------------+---------------+
# | Step Size  | Sequence Length|
# +------------+---------------+
# |     1      |       5       |
# |     7      |      18       |
# |    14      |      30       |
# |    30      |      70       |
# |    60      |     120       |
# +------------+---------------+
step_size = 
X_test_processed, y_test_processed = get_processed_test_data(X_test, y_test, step_size, sequence_length)

y_pred=model.predict(X_test_processed)

mse = mean_squared_error(y_test_processed, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

mae = mean_absolute_error(y_test_processed, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

y_test_values = [val[0] for val in y_test_processed]
y_pred_values = [val[0] for val in y_pred]
# Plotting the actual vs predicted values
plt.figure(figsize=(20, 5))
plt.plot(y_test_values, label=f'Actual (Step Size {step_size})', alpha=0.6)
plt.plot(y_pred_values, label=f'Predicted (Step Size {step_size})', alpha=0.6)

plt.title(f'Actual vs Predicted PM25 Concentration on Test Data (Step Size {step_size})')
plt.xlabel('Time Step')
plt.ylabel('Normalized PM25 Concentration')
plt.legend()
plt.show()

