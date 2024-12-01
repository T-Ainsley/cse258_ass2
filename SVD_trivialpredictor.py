
#%%
#Importing relevant packages
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import re
import scikitplot.metrics as skplt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
#%% md
# #**1. Data Cleaning**
#%%
data = pd.read_json('renttherunway_final_data.json', lines=True)
#%%
data.rename(columns={'bust size': 'bust_size', 'rented for': 'rented_for', 'body type':'body_type'}, inplace=True)

#%% md
# ##1.1 Dealing with missing values
#%%
data.isna().sum()
#%%
size_null = ['bust_size', 'weight', 'rented_for', 'body_type', 'height']
for i in size_null:
  i_sum = data.groupby(i).size().sort_values(ascending=False)
  print(i_sum.head(1))
#%%
### Use the most common value in each column to fill missing data
#bust size
data['bust_size'].fillna('34b', inplace=True)
#weight
data['weight'].fillna('130lbs', inplace=True)
#rating
data['rating'].fillna(float(data['rating'].median()), inplace=True)
#rented for
data['rented_for'].fillna('wedding', inplace=True)
#body type
data['body_type'].fillna('hourglass', inplace=True)
#height
data['height'].fillna('5\' 4"', inplace=True)
#age
data['age'].fillna(float(data['age'].median()), inplace=True)
#%%
data.isna().sum()

#%% md
# ##1.2 Converting data type
#%%
data['review_date'] = pd.to_datetime(data['review_date'])
data['weight'] = data.weight.str.replace('lbs', '')
#%%
data.head(3)
#%%
data['bust_size'].unique()
#%%
def bust_size_to_numeric(bust_size):
    if pd.isna(bust_size):
        print('nan')
        return np.nan
    try:
        band_size = int(re.search(r'\d+', bust_size).group())
        cup_part = re.sub(r'\d+', '', bust_size).lower()
        cup_mapping = {
            'aa': 0.5, 'a': 1, 'b': 2, 'c': 3, 'd': 4,
            'dd': 5, 'e': 5, 'ddd': 6, 'f': 6,
            'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11
        }
        if '+' in cup_part:
            cup_part = cup_part.replace('+', '')
            cup_value = cup_mapping.get(cup_part, 0) + 0.5
        elif '/' in cup_part:
            cup_part = cup_part.split('/')[0]
            cup_value = cup_mapping.get(cup_part, 0)
        else:
            cup_value = cup_mapping.get(cup_part, 0)
        return band_size + cup_value
    except:
        return np.nan

#%%
data['bust_size'] = data['bust_size'].apply(bust_size_to_numeric)
#%%
data['bust_size'].unique()
#%%
data.info()
#%%
data['weight'] = data['weight'].astype('float64')
#%%
def feet_to_float(cell_string):
    try:
        split_strings = cell_string.replace('"','').replace("'",'').split()
        float_value = float(split_strings[0])*12+float(split_strings[1])
    except:
        float_value = np.nan
    return float_value
#%%
data['height'] = data['height'].apply(feet_to_float)
#%%
### Filter the outliers
data = data[data.age <= 100]
data = data[data['size'] <= 22]
#%%
data.reset_index(drop=True,inplace= True)
#%%
data.height
#%%
data.info()
#%%
# Convert 'fit' column from boolean to 0/1/2
# Replace 'small' with 0, 'fit' with 1, and 'large' with 2
data['fit_encoded'] = data['fit'].map({'small': 0, 'fit': 1, 'large': 2})
# Ensure the column is correctly encoded as integers
data['fit_encoded'] = data['fit_encoded'].astype(int)
# Check the result
print(data[['fit', 'fit_encoded']].head())

# Average the ratings if there are duplicate user-item pairs
df = data.groupby(['user_id', 'item_id'], as_index=False)['fit_encoded'].agg(lambda x: x.mode()[0])
#%%
# Step 1: Prepare data
user_item_matrix = df.pivot(index='user_id', columns='item_id', values='fit_encoded')
# Split data into train and test
train_matrix = user_item_matrix.copy()
test_matrix = user_item_matrix.copy()

# Mask test set in training matrix
non_nan_mask = ~np.isnan(user_item_matrix.values)
test_mask = np.random.rand(*user_item_matrix.shape) < 0.2
final_test_mask = non_nan_mask & test_mask  # Only test on non-NaN values

train_matrix[final_test_mask] = np.nan  # Hide some test data in the training matrix

# Apply SVD to the training data only
mean_values = train_matrix.mean()
train_matrix_filled = train_matrix.apply(lambda x: x.fillna(x.mean()), axis=0)  # Handle missing values
print(train_matrix_filled.isna().sum())

# Calculate the mean of the entire dataset, ignoring NaN values
overall_mean = train_matrix.stack().mean()
# Replace columns with all NaN values with the mean of the entire dataset
train_matrix_filled = train_matrix_filled.apply(lambda x: x.fillna(overall_mean) if x.isna().all() else x, axis=0)
# Check the result
print(train_matrix_filled.isna().sum())  # Should show 0 NaNs for all columns
#%%
svd = TruncatedSVD(n_components=50)
train_svd = svd.fit_transform(train_matrix_filled)

# Predict/reconstruct the full matrix
train_reconstructed = np.dot(train_svd, svd.components_)

# Evaluate on the hidden test set
test_values = test_matrix.values[final_test_mask]
predicted_values = train_reconstructed[final_test_mask]
mse = mean_squared_error(test_values, predicted_values)
print(mse)

#%%
# Define bin edges corresponding to class boundaries
bins = [-np.inf, 0.5, 1.5, np.inf]  # Example bins for 'small', 'fit', 'large'
labels = [0, 1, 2]  # Class labels

# Assign discrete labels based on bins
y_pred_discrete = np.digitize(predicted_values, bins=bins) - 1
y_true_discrete = np.digitize(test_values, bins=bins) - 1
# baseline
# Calculate the global mean rating
global_mean = train_matrix_filled.mean().mean()
Y_pred = np.full_like(test_values, global_mean)
mse_item_mean = mean_squared_error(test_values, Y_pred)
print(f"Item Mean MSE: {mse_item_mean}")

# plott_confusion_matrix(y_true_discrete ,y_pred_discrete)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.show()
f1 = f1_score(y_true_discrete, y_pred_discrete, average='weighted')
print(f1)
#%%
# Calculate accuracy
accuracy = accuracy_score(y_true_discrete, y_pred_discrete)
print("Accuracy: ", accuracy)
mse = mean_squared_error(y_true_discrete, y_pred_discrete)
print("trivial predictor mse: ",mse)

