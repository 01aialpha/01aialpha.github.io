```python
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, precision_recall_fscore_support, f1_score

import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')


```


```python
data = pd.read_csv('Churn_Modelling.csv') #Kaggle Data
```


```python
data.head() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RowNumber</th>
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>15634602</td>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# 1. 데이터 탐색 및 Preprocessing
# (step1) 필요없는 변수 제거 : Surname

# (step2) 이산형과 연속형을 구분하고, 이산형은 다시 명목척도와 서열척도로 구분하여, 변수분류

1. 이산형_명목척도 : Geography(4cls), Age_1(5cls), Gender(2cls), HasCrCard(2cls), IsActiveMember(2cls)
2. 이산형_서열척도 : Tenure(11cls), NumofProducts(4cls), HasCrCard(2cls), IsActiveMember(2cls) 
3. 연속형 : CreditScore, Balance, EstimateSalary

# (step3) scaling
1. 이산형_명목척도 : dummy variable
2. 이산형_서열척도 : min_max scaling
3. 연속형 : normalization scaling

```python
# 컬럼삭제
data = data.drop(['Surname'], axis=1) 
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 13 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   RowNumber        10000 non-null  int64  
     1   CustomerId       10000 non-null  int64  
     2   CreditScore      10000 non-null  int64  
     3   Geography        10000 non-null  object 
     4   Gender           10000 non-null  object 
     5   Age              10000 non-null  int64  
     6   Tenure           10000 non-null  int64  
     7   Balance          10000 non-null  float64
     8   NumOfProducts    10000 non-null  int64  
     9   HasCrCard        10000 non-null  int64  
     10  IsActiveMember   10000 non-null  int64  
     11  EstimatedSalary  10000 non-null  float64
     12  Exited           10000 non-null  int64  
    dtypes: float64(2), int64(9), object(2)
    memory usage: 1015.8+ KB
    


```python
data.shape
```




    (10000, 13)




```python
data = data.set_index('RowNumber') 
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerId</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
    <tr>
      <th>RowNumber</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>15634602</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15647311</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15619304</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15701354</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15737888</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data1 = data.copy() 
data = data.drop(['Exited'], axis=1)
target = data1['Exited'] 
```

### AGE 클래스 변환


```python
data['Age_gbn'] = data['Age'].map(lambda x:int(x/10))
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerId</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Age_gbn</th>
    </tr>
    <tr>
      <th>RowNumber</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>15634602</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15647311</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15619304</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15701354</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15737888</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
age_gbn = data['Age_gbn'].value_counts()
```


```python
age_gbn.sort_index()
```




    1      49
    2    1592
    3    4346
    4    2618
    5     869
    6     375
    7     136
    8      13
    9       2
    Name: Age_gbn, dtype: int64




```python
data['Age_group'] = data['Age_gbn']
```


```python
for i in range(1,len(data)+1):
    if data.loc[i]['Age_gbn'] <= 2:
        data.loc[i,'Age_group'] = 2
    elif data.loc[i]['Age_gbn'] >= 6:
        data.loc[i,'Age_group'] = 6
    else :
        data.loc[i,'Age_group'] = data.loc[i,'Age_gbn']
```


```python
data['Age_group'].value_counts()
```




    3    4346
    4    2618
    2    1641
    5     869
    6     526
    Name: Age_group, dtype: int64




```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerId</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Age_gbn</th>
      <th>Age_group</th>
    </tr>
    <tr>
      <th>RowNumber</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>15634602</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15647311</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15619304</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15701354</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15737888</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.drop(['Age', 'Age_gbn'], axis=1, inplace=True)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerId</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Age_group</th>
    </tr>
    <tr>
      <th>RowNumber</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>15634602</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15647311</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15619304</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15701354</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15737888</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

# 범주형 변수 처리

LabelEncoder – for labels(response variable) coding 1,2,3… [implies order]

OrdinalEncoder – for features coding 1,2,3 … [implies order]

Label Binarizer – for response variable, coding 0 & 1 [ creating multiple dummy columns]

OneHotEncoder - for feature variables, coding 0 & 1 [ creating multiple dummy columns]

A quick example can be found https://colab.research.google.com/drive/1JAzJ2qcC2VnHoXCXNHo_ZKyAG4aW0QVU?usp=sharing.

### 1. Geography


```python
Geo_columns = pd.get_dummies(data['Geography'], drop_first=True) 
```


```python
Geo_columns
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Germany</th>
      <th>Spain</th>
    </tr>
    <tr>
      <th>RowNumber</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10000</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 2 columns</p>
</div>



#### 2. 성별


```python
# label_binarizer = preprocessing.LabelBinarizer()
# le=preprocessing.LabelEncoder()
# ohe=preprocessing.OneHotEncoder(sparse=False)
```


```python
#data['Gender'] = label_binarizer.fit_transform(data['Gender'].values) 
#data['Gender1'] = le.fit_transform(data['Gender'].values) 
#data['Gender'] = ohe.fit_transform(data['Gender'].values.reshape(-1,1)).astype(int)
```


```python
Gen_columns = pd.get_dummies(data['Gender'], drop_first=True) 
```


```python
dataset = data.join(Geo_columns)
dataset= dataset.join(Gen_columns)
```


```python
data = dataset.drop(columns=['Geography', 'Gender'])
data.head() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerId</th>
      <th>CreditScore</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Age_group</th>
      <th>Germany</th>
      <th>Spain</th>
      <th>Male</th>
    </tr>
    <tr>
      <th>RowNumber</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>15634602</td>
      <td>-0.326221</td>
      <td>0.2</td>
      <td>-1.225848</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0.021886</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15647311</td>
      <td>-0.440036</td>
      <td>0.1</td>
      <td>0.117350</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>0.216534</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15619304</td>
      <td>-1.536794</td>
      <td>0.8</td>
      <td>1.333053</td>
      <td>0.666667</td>
      <td>1</td>
      <td>0</td>
      <td>0.240687</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15701354</td>
      <td>0.501521</td>
      <td>0.1</td>
      <td>-1.225848</td>
      <td>0.333333</td>
      <td>0</td>
      <td>0</td>
      <td>-0.108918</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15737888</td>
      <td>2.063884</td>
      <td>0.2</td>
      <td>0.785728</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>-0.365276</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# 스케일링


```python
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
std_scaler = preprocessing.StandardScaler()

```


```python
data['CreditScore'] = std_scaler.fit_transform(data['CreditScore'].values.reshape(-1,1)) 
data['Balance'] = std_scaler.fit_transform(data['Balance'].values.reshape(-1,1))
data['EstimatedSalary'] = std_scaler.fit_transform(data['EstimatedSalary'].values.reshape(-1,1))
```


```python
data['Tenure'] = min_max_scaler.fit_transform(data['Tenure'].values.reshape(-1,1)) 
data['NumOfProducts'] = min_max_scaler.fit_transform(data['NumOfProducts'].values.reshape(-1,1))
```


```python
data.head() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerId</th>
      <th>CreditScore</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Age_group</th>
      <th>Germany</th>
      <th>Spain</th>
      <th>Male</th>
    </tr>
    <tr>
      <th>RowNumber</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>15634602</td>
      <td>-0.326221</td>
      <td>0.2</td>
      <td>-1.225848</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>0.021886</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15647311</td>
      <td>-0.440036</td>
      <td>0.1</td>
      <td>0.117350</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>0.216534</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15619304</td>
      <td>-1.536794</td>
      <td>0.8</td>
      <td>1.333053</td>
      <td>0.666667</td>
      <td>1</td>
      <td>0</td>
      <td>0.240687</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15701354</td>
      <td>0.501521</td>
      <td>0.1</td>
      <td>-1.225848</td>
      <td>0.333333</td>
      <td>0</td>
      <td>0</td>
      <td>-0.108918</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15737888</td>
      <td>2.063884</td>
      <td>0.2</td>
      <td>0.785728</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>-0.365276</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerId</th>
      <th>CreditScore</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Age_group</th>
      <th>Germany</th>
      <th>Spain</th>
      <th>Male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>10000.000000</td>
      <td>1.000000e+04</td>
      <td>10000.000000</td>
      <td>10000.00000</td>
      <td>10000.000000</td>
      <td>1.000000e+04</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.569094e+07</td>
      <td>-2.766676e-17</td>
      <td>0.501280</td>
      <td>3.698153e-17</td>
      <td>0.176733</td>
      <td>0.70550</td>
      <td>0.515100</td>
      <td>-4.085621e-18</td>
      <td>3.429300</td>
      <td>0.250900</td>
      <td>0.247700</td>
      <td>0.545700</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.193619e+04</td>
      <td>1.000050e+00</td>
      <td>0.289217</td>
      <td>1.000050e+00</td>
      <td>0.193885</td>
      <td>0.45584</td>
      <td>0.499797</td>
      <td>1.000050e+00</td>
      <td>1.030877</td>
      <td>0.433553</td>
      <td>0.431698</td>
      <td>0.497932</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.556570e+07</td>
      <td>-3.109504e+00</td>
      <td>0.000000</td>
      <td>-1.225848e+00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>-1.740268e+00</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.562853e+07</td>
      <td>-6.883586e-01</td>
      <td>0.300000</td>
      <td>-1.225848e+00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>-8.535935e-01</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.569074e+07</td>
      <td>1.522218e-02</td>
      <td>0.500000</td>
      <td>3.319639e-01</td>
      <td>0.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.802807e-03</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.575323e+07</td>
      <td>6.981094e-01</td>
      <td>0.700000</td>
      <td>8.199205e-01</td>
      <td>0.333333</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>8.572431e-01</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.581569e+07</td>
      <td>2.063884e+00</td>
      <td>1.000000</td>
      <td>2.795323e+00</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.737200e+00</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



# 2. 머신러닝


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state=0)
```


```python
X_train.shape
```




    (8000, 12)



### Logistic Regression (베이스라인 모델!!)


```python
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```


```python
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train) 
```




    LogisticRegression(random_state=42)




```python
y_pred = log_reg.predict(X_test)
y_pred
```




    array([0, 0, 0, ..., 0, 0, 0], dtype=int64)




```python
accuracy = ((y_test==y_pred).sum() / len(predY))*100
print("정확도 = %.2f" % accuracy, "%")
```

    정확도 = 79.75 %
    


```python
accuracy_score(y_pred, y_test) 
```




    0.7975




```python
precision_score(y_pred, y_test)
```




    0.0




```python
recall_score(y_pred, y_test)
```




    0.0




```python
f1_score(y_pred, y_test)
```




    0.0


