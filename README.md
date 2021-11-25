# Manifold Take Home Problem

Usage:
To run all three modules on default, use the commands:

```python
  python preprocessing.py
  python train.py
  python evaluate.py
```

from root.

To use different input files or to change hyperparameters and control options, use the available run-time flags. For help, run with `-h` i.e. `python preprocessing.py -h`.

Each module will output storage files to the root directory for the next module to use. To save files for future use, rename or move them and change the run-time file input flags to reference them.

## Changes that I would make with more time
I limited myself to a few hours for this challenge, but with more time I would refine more hyperparameters by performing a grid search with several potential hyperparameter inputs using an algorithm such as GridSearchCV. I would also implement unit tests. I would test preprocessing.py by creating an example small dataset and ensuring that it was properly separated into the correct sized outputs. I would test train.py by training an extremely simple dataset for a random forest to model (probably something like a feature ranging from 1:10 with 0's for low numbers and 1's for high numbers) and verifying that the program easily regresses the data. I would test evaluate.py by ensuring that the MAE and RMSE are correct for a known set of data.

I would also implement a job name system, where the user can specify job name, and the program will automatically use this name to find the input files and save output files. I also might write scripts so that the user doesn't have to directly deal with the python commands, and can simply run the scripts with the given flags from the root directory.
