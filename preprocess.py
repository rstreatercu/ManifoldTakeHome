import argparse
import pandas as pd
import numpy as np

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_file', type=str, nargs=1, help='custom data file', default="home_data.csv")
parser.add_argument('--train_percent', type=float, nargs=1, help='percent of input to use for training', default=[50.0])
parser.add_argument('-f', '--feature_file', type=str, nargs=1, help='names of features to consider in analysis', default="defaults/features.csv")
parser.add_argument('-t', '--test', help='run on unit test mode', action='store_true')
args = parser.parse_args()

try:
    data = pd.read_csv(args.data_file)
    # Save columns to array in order to specify data columns to use.
    columns = np.array(pd.read_csv(args.feature_file).columns)
except:
    print("Bad file inputs. Run with -h flag for help menu.")
# Only save data columns specified, plus price because it's the target
data = data[np.append(columns,"price")]
data_train = data.sample(frac = args.train_percent[0]/100.0)

# Save the rest of the data to evaluate
eval_index = np.full(len(data.index),True,dtype=bool)
eval_index[data_train.index] = False
data_evaluate = data[eval_index]

data_evaluate.to_csv("data_evaluate.csv",index=False)
data_train.to_csv("data_train.csv",index=False)
data_train.to_csv()