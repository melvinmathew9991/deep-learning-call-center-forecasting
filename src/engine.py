import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ml_pipeline.cnn_lstm_model import CNN_LSTM
from ml_pipeline.cnn_model import CNN_model
from ml_pipeline.lstm_model import LSTM_model
from ml_pipeline.mlp import MLP
from ml_pipeline.utils import DATA_DIR, MODELS_DIR, OUTPUT_DIR, PLOTS_DIR, seed_everything

warnings.filterwarnings("ignore")
seed_everything()

# importing the data
raw_csv_data = pd.read_excel(DATA_DIR / "CallCenterData.xlsx")

# check point of data
df_comp = raw_csv_data.copy()

# taken as a date time field
df_comp.set_index("month", inplace=True)

# seeting the frequency as monthly
df_comp = df_comp.asfreq('M')


### Time Series Visualization ###

df_comp.Healthcare.plot(figsize=(20,5), title="Healthcare")
plt.savefig(PLOTS_DIR / "healthcare.png")

df_comp.Telecom.plot(figsize=(20,5), title="Telecom")
plt.savefig(PLOTS_DIR / "telecom.png")

df_comp.Banking.plot(figsize=(20,5), title="Banking")
plt.savefig(PLOTS_DIR / "banking.png")

df_comp.Technology.plot(figsize=(20,5), title="Technology")
plt.savefig(PLOTS_DIR / "tech.png")

df_comp.Insurance.plot(figsize=(20,5), title="Insurance")
plt.savefig(PLOTS_DIR / "insurance.png")


### Setting the training format  ###

data = df_comp.Healthcare

#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
#In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training).

#Empty lists to be populated using formatted training data
target_data = []

# Number of days we want to look into the future based on the past days.
n_past = 5  # Number of past days we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (?)
#refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(len(data)):
    temp = []
    for j in range(n_past + 1):
        try:
            temp.append(data[i+j])
        except IndexError:
            continue
    if len(temp) > 5:
        target_data.append(temp)

len(target_data)


### Train / Validation / Test Split ###

data_df = pd.DataFrame(target_data, columns=["t-4","t-3","t-2","t-1","t","Y"]) # create a dataframe
##data_df.head()


# test set split (chronological: held-out, most recent 20 points)
test_size = 20
train = data_df[:-test_size]
test = data_df[-test_size:]

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset - fit only on the training split, then apply to both splits
scaler = MinMaxScaler(feature_range=(0, 1)) # define min max scaler
scaler = scaler.fit(train)                  # fit the scaler on train data only
train = pd.DataFrame(scaler.transform(train), columns=data_df.columns) # transform the train data
test = pd.DataFrame(scaler.transform(test), columns=data_df.columns)   # transform the test data using the train-fitted scaler

X, Y = train.drop("Y", axis=1).values, train["Y"].values # drop the column

# validation set split (chronological: most recent slice of what remains after removing test)
valid_fraction = 0.1
n_valid = max(1, round(len(X) * valid_fraction))
X_train, X_valid = X[:-n_valid], X[-n_valid:]
Y_train, Y_valid = Y[:-n_valid], Y[-n_valid:]

X_test, Y_test = test.drop("Y", axis=1).values, test["Y"]


### Build the models ###
mlp_model = MLP(X_train, X_valid, X_test, Y_train, Y_valid, Y_test)
mlp_model.model.save(MODELS_DIR / "mlp_model.keras")


cnn_model = CNN_model(X_train, X_valid, X_test, Y_train, Y_valid, Y_test)
cnn_model.model.save(MODELS_DIR / "cnn_model.keras")


lstm_model = LSTM_model(X_train, X_valid, X_test, Y_train, Y_valid, Y_test)
lstm_model.model.save(MODELS_DIR / "lstm_model.keras")


cnn_lstm_model = CNN_LSTM(X_train, X_valid, X_test, Y_train, Y_valid, Y_test)
cnn_lstm_model.model.save(MODELS_DIR / "cnn_lstm_model.keras")


### Model comparison summary ###

results = pd.DataFrame([
    mlp_model.metrics,
    cnn_model.metrics,
    lstm_model.metrics,
    cnn_lstm_model.metrics,
]).sort_values("rmse").reset_index(drop=True)

print("\nModel comparison (sorted by RMSE, on scaled test data):")
print(results.to_string(index=False))
results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
