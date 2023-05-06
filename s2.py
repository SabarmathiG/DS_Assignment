import pandas as pd
import json

# load JSON data from file
with open('algoparams_from_ui') as f:
    data = json.load(f)

# load dataset
df = pd.read_csv('iris.csv')

#list of features
features=["sepal_length","sepal_width","petal_length","petal_width"]

d=data["design_state_data"]["feature_handling"]

for feature in features:
    if d[feature]['feature_details']['missing_values'] == 'Impute':
        #impute method and value
        impute_with = d[feature]['feature_details']['impute_with']
        impute_value = d[feature]['feature_details']['impute_value']

        if impute_with == "Average of values":
            df[feature].fillna(impute_value, inplace=True)
        elif impute_with == "custom":
            df[feature].fillna(impute_value, inplace=True)
#save
df.to_csv('iris.csv', index=False)
     



