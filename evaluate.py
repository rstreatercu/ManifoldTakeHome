import argparse
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--evaluate_data', type=str, nargs=1, help='evaluation data file', default="data_evaluate.csv")
parser.add_argument('-r', '--regressor_file', type=str, nargs=1, help='regressor variable file', default="regressor.joblib")
parser.add_argument('-t', '--test', help='run on unit test mode', action='store_true')

args = parser.parse_args()

try:
    data = pd.read_csv(args.evaluate_data)
    regressor = joblib.load(args.regressor_file)
    target = pd.Series(data["price"])
    features = data.drop("price",axis='columns')
except:
    print("Bad input file(s). Run with -h flag for help menu.")

prediction = regressor.predict(features)
plt.scatter(target.index[1:100],target[1:100])
plt.scatter(target.index[1:100],prediction[1:100])
plt.show()