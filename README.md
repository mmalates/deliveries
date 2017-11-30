# Delivery Time Predictor

When a delivery order comes in we need to give an accurate prediction of how long it will take for the order to arrive at the customer's location.  I built an automated pipeline so that a model could be trained on any data set with the same set of features.  The model is then used in an application to predict the delivery time for incoming delivery requests.

## Getting Started

Upload your training data as a comma separated csv or json into a folder called 'data' in the root directory.  Then run ```python train.py <input_file>``` from the ```src``` directory.  The training program will output a file called ```rf.dill``` to the ```app``` directory.  Upload your data to predict as a comma separated csv or json to the ```app``` folder with an added column ```delivery_id```.  Run ```python predict.py <data_to_predict>```.  The output will be written with tab separation in ```predictions.csv```.

### Required Packages

```
Python 2.7
Scikit-Learn
Pandas
Dill
```