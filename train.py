import argparse
import pandas as pd

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--train_data', type=str, nargs=1, help='training data file', default="data_train.csv")
parser.add_argument('-t', '--test', help='run on unit test mode', action='store_true')
args = parser.parse_args()

try:
    data = pd.read_csv(args.train_data)
    
    # Cost is the target variable; all else are features.
    target = pd.Series(data["price"])
    features = data.drop("price",axis='columns')
except:
    print("Bad training data file. Run with -h flag for help menu.")

