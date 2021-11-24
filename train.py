import argparse
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--train_data', type=str, nargs=1, help='training data file', default="data_train.csv")
parser.add_argument('-t', '--test', help='run on unit test mode', action='store_true')
parser.add_argument('-n','--n_trees', type=int, nargs=1, help='number of decision trees to use', default=[50])

args = parser.parse_args()

try:
    data = pd.read_csv(args.train_data)
    
    # Cost is the target variable; all else are features.
    target = pd.Series(data["price"])
    features = data.drop("price",axis='columns')
except:
    print("Bad training data file. Run with -h flag for help menu.")

regressor = RandomForestRegressor(n_estimators = args.n_trees[0])
regressor.fit(features,target)

joblib.dump(regressor,"regressor.joblib")