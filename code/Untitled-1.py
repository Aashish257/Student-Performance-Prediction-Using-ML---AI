# %%

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# %%
df_mat=pd.read_csv('Dataset/student-mat.csv',sep=';')


# %%
df_mat

# %%
df_mat.info()

# %%
df_mat.columns

# %%
for i in df_mat.columns:
    print(df_mat[i].value_counts())

# %% [markdown]
# # Data is imbalanced, We will use Tree based models

# %% [markdown]
# # Removing Outliers,if any

# %%
result = df_mat.select_dtypes(include='number')

# %%
for i in result.columns:
    percentile25 = df_mat[i].quantile(0.25)
    percentile75 = df_mat[i].quantile(0.75)
    
    iqr = percentile75-percentile25
    
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    
    df_mat[df_mat[i] > upper_limit]
    df_mat[df_mat[i] < lower_limit]
    
    df = df_mat[df_mat[i] < upper_limit ]
    df = df_mat[df_mat[i] > lower_limit ]
    


# %%
df

# %%
df_mat.columns

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
X=df.drop('G3',axis=1) # Independent Variables
Y=df.G3 # Dependent variable

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = .25, random_state = 111)

# %%
X.columns

# %% [markdown]
# # Defining Models

# %%
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
RF=RandomForestRegressor(max_depth=100)
model=RF.fit(x_train,y_train)

# %%
import pickle
import pickle
pickle.dump(model, open('model.sav', 'wb'))

# %%
y_pred=model.predict(x_test) # Predicted Output

# %%
from sklearn.metrics import explained_variance_score

explained_variance_score( y_pred,y_test) # Y_pred ->> Predicted label / Y_test ->> True label

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
    x[7] =paid
    x[8] = higher 
    x[9] = internet
    x[10] =  goout
    x[11] =  G1
    x[12] =  G2
    
    
    
    return model.predict([x])[0]

# %%
predict_price(1,17, 1, 1,1,1, 1,
       0,1, 1,4, 4,1)

# %%



