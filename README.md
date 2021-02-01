# Comparison of Outlier Detection Algorithms on NILM Datasets
This repository contains the simulation script used for the bachelor thesis "Comparison of Outlier Detection Algorithms in NILM Datasets". It also contains the corresponding Conda environment and the results of the simulations.

## Experiment setup:
- 5-fold cross-validation
- No filter, rolling median and hampel filter
- Time period of the REFIT data: 2013-10-09 - 2014-04-08

## Usage:

1. Create conda environment with `conda env create -f environment. yml`
2. Run script with `python outlierMain.py sys.argv` e.g. `python outlierMain.py REFIT.h5 'dish washer' '2013-10-09' '2014-04-08'`

| sys.argv      | Usage           | Description  |
| ------------- |:-------------:| :-----|
| `sys.argv[1]` | h5File                  | Either the absolute path to the file or if file in h5Files directory just the filename e.g.: REFIT.h5 |
| `sys.argv[2]` | applications/aggregate  | **Applications**: The applications to be examined as a string separated by a comma e.g.: `'kettle,computer'` <br /> **Aggregate**: Pass `'aggregate'` if the aggregate data of all 20 houses should be used  |
| `sys.argv[3]` | windowStart             | Start time of the window: e.g.: `'2013-09-06'` or `None` to use the whole dataset  |
| `sys.argv[4]` | windowEnd               | End time of the window: e.g.: `'2013-09-08'` |
