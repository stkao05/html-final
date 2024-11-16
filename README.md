## Overview

`data/`: competition data, unmodified

`/data_loader.py`: a script that feature engineers the competition data (e.g. filling missing value) so that model could better fit the data. 

`data_processed/`: datas that have been processed. It should have no missing value in its numerical column (but still could have missing value in its categorical col). The detail of processing method could be view at `/data_loader.py`


## Setup

```
pip install -r requirements.txt
```

## Test

```
pytest test_data_loader.py
```


## Benchmark

### Test 1

| Algorithm   | Feature engineering    | Validation method                      | Validation acc | Test acc (submission) |
|-------------|------------------------|----------------------------------------|----------------|------------------------|
| logistic, with C=0.001    | using /data_processed data  | 5 CV                                  | 0.5588         | 0.57650                |
| LGBM        | see `lgbm.ipynb`       | train: Jan-June, validation: July     | 0.5671         | 0.59554                |


### Test 2

| Algorithm   | Feature engineering    | Validation method                      | Validation acc | Test acc (submission) |
|-------------|------------------------|----------------------------------------|----------------|------------------------|
| logistic, with C=0.001    | using /data_processed data  | 5 CV                                  | 0.5588         | 0.58222                |
