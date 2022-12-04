# %% [markdown]
# # Importing libraries

# %%

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# %% [markdown]
# # Reading dataset

# %%
df_mat=pd.read_csv('Dataset\student-mat.csv',sep=';')


# %%
df_mat

# %% [markdown]
# # Preprocessing

# %% [markdown]
# * Information

# %%
df_mat.info()

# %% [markdown]
# * Columns

# %%
df_mat.columns

# %% [markdown]
# * Value of each category in a column

# %%
for i in df_mat.columns:
    print(df_mat[i].value_counts())

# %% [markdown]
# # Data is imbalanced, We will use Tree based models

# %% [markdown]
# * Removing Outliers,if any

# %%
result = df_mat.select_dtypes(include='number')

# %%
for i in result.columns:
    percentile25 = df_mat[i].quantile(0.25) # 25 th percentile of my data
    percentile75 = df_mat[i].quantile(0.75) # 75 th percentile of my data
    
    iqr = percentile75-percentile25 # Defined a range
    
    upper_limit = percentile75 + 1.5 * iqr # Defining Upper Limit
    lower_limit = percentile25 - 1.5 * iqr # Defining Lower Limit
    
    df_mat[df_mat[i] > upper_limit]
    df_mat[df_mat[i] < lower_limit]
    
    df = df_mat[df_mat[i] < upper_limit ] # datapoints less than my upper limit
    df = df_mat[df_mat[i] > lower_limit ] # datapoints greater than my lower limit
    


# %%
df

# %%
df_mat.columns

# %% [markdown]
# # Obj dtype to numeric dtype

# %%
col=['schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic']
dic={'no':0,'yes':1}

for i in col:
    
    df_mat[i]=df_mat[i].map(dic)

# %%
df.corr()['G3'].sort_values()

# %% [markdown]
# * schoolsup    -0.082788
# * health       -0.061335
# * Dalc         -0.054660
# * Walc         -0.051939
# * famsup       -0.039157
# * freetime      0.011307
# 
# **These features are having very less correlation, Lets drop them**

# %%
df.drop(columns=['schoolsup','health','Dalc','Walc','famsup','freetime'],inplace=True)

# %%
df.drop(columns=['nursery','romantic'],inplace=True)

# %%
df

# %% [markdown]
# 1. **See value counts of F job and M job ,'other' is almost equal or greater than rest of value. and here other is generic value. For betterment lets drop these columns**
# 
# 2. **We should also drop school as out of 2 school, 1 has 70% more enrollement**
# 

# %%
df=df.drop(columns=['Mjob','Fjob','school'])

# %%
df.reason.value_counts()

# %%
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df.address=le.fit_transform(df.address)
df.sex=le.fit_transform(df.sex)
df.famsize=le.fit_transform(df.famsize)
df.Pstatus=le.fit_transform(df.Pstatus)
df.paid=le.fit_transform(df.paid)
df.activities=le.fit_transform(df.activities)
df.higher=le.fit_transform(df.higher)
df.internet=le.fit_transform(df.internet)


# %%
for i in df_mat.columns:
    print(df_mat[i].value_counts())

# %%
df.guardian=le.fit_transform(df.guardian)

# %%
df.corr()['G3'].sort_values()

# %% [markdown]
# * Correlation

# %%
df.drop(columns=['studytime','famsize','famrel','absences','activities','Pstatus','guardian'],inplace=True)

# %%
df

# %%
## one hot encoding the 'Location' column
dummies = pd.get_dummies(df['reason'])
df = pd.concat([df,dummies], axis='columns')
df.drop('reason', axis=1, inplace=True)

# %%
df.corr()['G3'].sort_values()

# %%
df.drop(columns=['course','home','other','reputation'],inplace=True)

# %%
from sklearn.model_selection import train_test_split
X=df.drop('G3',axis=1)
Y=df.G3

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = .25, random_state = 111)

# %%
x_train.shape

# %%
X.columns

# %% [markdown]
# # Random Forest

# %%
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
RF=RandomForestRegressor(max_depth=15)
model=RF.fit(x_train,y_train)

# %%

import pickle
pickle.dump(model, open('model.sav', 'wb')) #wb is write in binary

# %%
y_pred=model.predict(x_test)

# %%
y_test = y_test.values

# %%
from matplotlib import pyplot
errors = list()
for i in range(len(y_test)):
	# calculate error
	err = (y_test[i] - y_pred[i])**2
	# store error
	errors.append(err)
	# report error
	print('>%.1f, %.1f = %.3f' % (y_test[i], y_pred[i], err))
# plot errors
pyplot.plot(errors)
pyplot.xticks(ticks=[i for i in range(len(errors))], labels=y_pred)
pyplot.xlabel('y_pred Value')
pyplot.ylabel('Mean Squared Error')
pyplot.show()

# %%
from sklearn.metrics import explained_variance_score

explained_variance_score( y_pred,y_test)

# %% [markdown]
# # PREDICTION

# %%
def predict_price(sex, age, address, Medu, Fedu, traveltime, failures,
       paid, higher, internet, goout, G1, G2):
   
    x = np.zeros(len(X.columns))
    x[0] = sex 
    x[1] = age
    x[2] = address 
    x[3] = Medu 
    x[4] = Fedu
    x[5] = traveltime 
    x[6] = failures
    x[7] = paid
    x[8] = higher 
    x[9] = internet
    x[10] =  goout
    x[11] =  G1
    x[12] =  G2
    
    
    
    return model.predict([x])[0]

# %%
predict_price(1,17, 1, 1,1,1, 1,
       0,1, 1,4, 4,1)

# %% [markdown]
# # KNN

# %%
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=5,metric='minkowski',p=2)
model=model.fit(x_train,y_train)

# %%
y_pred=model.predict(x_test)

# %%
from sklearn.metrics import explained_variance_score

explained_variance_score( y_pred,y_test)

# %%
accuracy = model.score(x_test, y_test)
accuracy

# %%
y_test 

# %%
from matplotlib import pyplot
errors = list()
for i in range(len(y_test)):
	# calculate error
	err = (y_test[i] - y_pred[i])**2
	# store error
	errors.append(err)
	# report error
	print('>%.1f, %.1f = %.3f' % (y_test[i], y_pred[i], err))
# plot errors
pyplot.plot(errors)
pyplot.xticks(ticks=[i for i in range(len(errors))], labels=y_pred)
pyplot.xlabel('y_pred Value')
pyplot.ylabel('Mean Squared Error')
pyplot.show()

# %% [markdown]
# # XG B

# %%
import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

# %%
xg_reg.fit(x_train,y_train)

preds = xg_reg.predict(x_test)

# %%
from sklearn.metrics import explained_variance_score

explained_variance_score( y_pred,y_test)

# %%
accuracy = model.score(x_test, y_test)
accuracy

# %%
from matplotlib import pyplot
errors = list()
for i in range(len(y_test)):
	# calculate error
	err = (y_test[i] - y_pred[i])**2
	# store error
	errors.append(err)
	# report error
	print('>%.1f, %.1f = %.3f' % (y_test[i], y_pred[i], err))
# plot errors
pyplot.plot(errors)
pyplot.xticks(ticks=[i for i in range(len(errors))], labels=y_pred)
pyplot.xlabel('y_pred Value')
pyplot.ylabel('Mean Squared Error')
pyplot.show()

# %%


from keras.models import Sequential
from keras.layers import Dense

# create ANN model
modelANN = Sequential()

# Defining the Input layer and FIRST hidden layer, both are same!
modelANN.add(Dense(units=5, input_dim=13, kernel_initializer='normal', activation='relu'))

# Defining the Second layer of the model
# after the first layer we don't have to specify input_dim as keras configure it automatically
modelANN.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))

# The output neuron is a single fully connected node 
# Since we will be predicting a single number
modelANN.add(Dense(1, kernel_initializer='normal'))




# %%
# Compiling the model
modelANN.compile(loss='mean_squared_error', optimizer='adam')

# Fitting the ANN to the Training set
modelANN.fit(x_train, y_train ,batch_size = 20, epochs = 500, verbose=2)

# %%
y_pred_ANN = model.predict(x_test)

# %%
explained_variance_score( y_pred_ANN,y_test)


