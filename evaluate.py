"""
Short description - This module contains code to evaluate a model

Richelle Streater
"""

if __name__ == "__main__":
    import argparse
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import joblib
    from sklearn.ensemble import RandomForestRegressor

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--evaluate_data', type=str, help='evaluation data file', default="data_evaluate.csv")
    parser.add_argument('-r', '--regressor_file', type=str, help='regressor variable file', default="regressor.joblib")
    parser.add_argument('--n_plot', type=int, help='number of points to plot', default=100)
    #parser.add_argument('-t', '--test', help='run on unit test mode', action='store_true')

    args = parser.parse_args()

    try:
        data = pd.read_csv(args.evaluate_data)
        regressor = joblib.load(args.regressor_file)
        target = pd.Series(data["price"])
        features = data.drop("price",axis='columns')
    except:
        print("Bad input file(s). Run with -h flag for help menu.")

    prediction = regressor.predict(features)

    # Print out results
    print("Mean average error = "+str(np.mean(abs(prediction-target))))
    print("Root mean square error = "+str(np.sqrt(np.sum((prediction-target)**2)/len(target))))

    # Plot results
    plt.plot(target.index[1:args.n_plot],target[1:args.n_plot])
    plt.scatter(target.index[1:args.n_plot],prediction[1:args.n_plot])
    plt.legend(["Target","Prediction"])
    plt.ylabel("Price (dollars)")
    plt.show()