
<!-- [![PyPI version](https://badge.fury.io/py/sentimentanalyser.svg)](https://badge.fury.io/py/sentimentanalyser)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HitCount](http://hits.dwyl.io/ashhadulislam/sentiment-analyser-lib.svg)](http://hits.dwyl.io/ashhadulislam/sentiment-analyser-lib)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/sentimentanalyser.svg)](https://img.shields.io/pypi/dm/sentimentanalyser.svg)
[![CodeFactor](https://www.codefactor.io/repository/github/ashhadulislam/sentiment-analyser-lib/badge/master)](https://www.codefactor.io/repository/github/ashhadulislam/sentiment-analyser-lib/overview/master) -->
# KNNOR: K-Nearest Neighbor OveRsampling method

### About
An oversampling technique for imbalanced datasets.

### Installation

Use below command to install 

`pip install knnor-reg`


### Source

The folder knnor_reg contains the source code.



### Usage


Convert your dataset to numpy array.

All values of the data must be numeric.

The last column must be the target value

Function call: 5 inputs
```
augment(data=df.values,k=k,class_ind=1,N=45000,randmx=randmx)
```
- data is the array like input of data, last column of data is class label	
- k is number of neighbors, it should be bigger or equal to 1
- class_ind is the value of data that needs to be augmented. For example, if the class labels are 0 or 1 and the datapoints for 0 need to be upsampled, class_ind=0
- N is the number of Datapoints that needs to be added
- randmx will be a value between 0 and 1, inclusive. smaller the randmx, closer is the data to each original data. randmx, uniform[0,randmx], ; randmx<=1

The outputs are:

- Data_a: complete data with augmented datapoints

- Ext_d: Only the augmented data points

- Ext_not: The datapoints that was created but ignored

Example implementation
```
from knnor_reg import data_augment
knnor_reg=data_augment.KNNOR_Reg()
data=pd.read_csv("knnor_reg/concrete.csv")
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values    
X_new,y_new,_,_=knnor_reg.fit_resample(X,y)
print("Before",X.shape,y.shape)
print("After",X_new.shape,y_new.shape)
```


### Examples

Go to example folder to see a jupyter notebook with the implementation




### Read the docs
The documentation of the library is present at



### Citation
If you are using this library in your research please cite the following.

Ashhadul Islam, Samir Brahim Belhaouari, Atiq Ur Rahman, Halima Bensmail,
KNNOR: An oversampling technique for imbalanced datasets,
Applied Soft Computing,
2021,
108288,
ISSN 1568-4946,
https://doi.org/10.1016/j.asoc.2021.108288.

(https://www.sciencedirect.com/science/article/pii/S1568494621010942)



### Authors
- Ashhadul Islam: ashhadulislam@gmail.com, aislam@mail.hbku.edu.qa
- Dr Samir Brahim Belhaouari: samir.brahim@gmail.com, sbelhaouari@hbku.edu.qa
