import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV


data=sns.load_dataset("titanic")

# drop all rows with a nan-entry
data=data.dropna()

## select labels and features

labels=data.survived
features=data.iloc[:,1:]

# scale or hot encode features

features_categrorical=features.select_dtypes(exclude='number')
features_categrorical=pd.get_dummies(features_categrorical).reset_index(drop=True)

features_numeric=features.select_dtypes(include='number') ## Note: This will also scale bool variables!
features_num_columns=list(features_numeric.columns)

scaler=StandardScaler()
features_numeric=scaler.fit_transform(features_numeric)
features_numeric=pd.DataFrame(features_numeric, columns=features_num_columns).reset_index(drop=True)

features= pd.concat([features_numeric,features_categrorical], axis=1)

#split data and train model

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

# Hyperopt with gridsearch

#do a gridsearch

estimator=RandomForestClassifier()
parameters_RFC={'max_depth':[None, 1,2,3,4,5,10,20],  'criterion': ['gini', 'entropy'], 'bootstrap':[True, False], 'n_estimators':[1,2,5,10,20,50,100]}
model_grid=GridSearchCV(estimator,param_grid=parameters_RFC,cv=10, verbose=4, n_jobs=2)

model_grid.fit(X_train,y_train)

res_df=pd.DataFrame(model_grid.cv_results_)

res_df.to_csv("titanic_rf.csv")