import argparse
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--train_data', type=str, nargs=1, help='training data file', default="data_train.csv")
parser.add_argument('-t', '--test', help='run on unit test mode', action='store_true')
parser.add_argument('-n','--n_trees', type=int, help='number of decision trees to use', default=100)
parser.add_argument('--max_depth', type=int, help='max tree depth', default=10)
parser.add_argument('--min_samples_split', type=int, help='minimum samples required to split a node', default=10)
parser.add_argument('--min_samples_leaf', type=int, help='minimum samples on a leaf', default=5)
parser.add_argument('--bootstrap', type=bool, help='use bootstrap samples', default=True)
parser.add_argument('--max_features', type=str, choices=["auto","sqrt","log2"], help='how to determine maximum number of features', default='auto')
parser.add_argument('--criterion', type=str, choices=["squared_error", "absolute_error", "poisson"], help='which error type to split on', default='squared_error')

args = parser.parse_args()

try:
    data = pd.read_csv(args.train_data)
    
    # Cost is the target variable; all else are features.
    target = pd.Series(data["price"])
    features = data.drop("price",axis='columns')
except:
    print("Bad training data file. Run with -h flag for help menu.")

# Create and fit regressor
regressor = RandomForestRegressor(n_estimators=args.n_trees, max_depth=args.max_depth,
                                  min_samples_split=args.min_samples_split, min_samples_leaf=args.min_samples_leaf,
                                  bootstrap=args.bootstrap, max_features=args.max_features, criterion=args.criterion)
regressor.fit(features,target)

# Save to file
joblib.dump(regressor,"regressor.joblib")