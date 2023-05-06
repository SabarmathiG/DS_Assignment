import pandas as pd
import json
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

iris = pd.read_csv('iris.csv')

#target variable column
le = LabelEncoder()
iris['species'] = le.fit_transform(iris['species'])

# separate the features and target variable
X = iris.drop('species', axis=1)
y = iris['species']

with open('algoparams_from_ui') as f:
    data = json.load(f)
df = pd.read_csv('iris.csv')

feature_reduction_method=data["design_state_data"]["feature_reduction"]["feature_reduction_method"]
num_of_features_to_keep=data["design_state_data"]["feature_reduction"]["num_of_features_to_keep"]
depth_of_trees=data["design_state_data"]["feature_reduction"]["depth_of_trees"]
num_of_trees=data["design_state_data"]["feature_reduction"]["num_of_trees"]


iris = load_iris()


if feature_reduction_method == "No Reduction":
    print(df.head())
elif feature_reduction_method == "Correlation with target":
    X = iris.data
    y = iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = y
    corr = df.corr()['target'].copy()
    corr.sort_values(ascending=False, inplace=True)
    features_to_keep = list(corr.index[1:num_of_features_to_keep+1])
    print(df[features_to_keep + ['target']])
elif feature_reduction_method == "Tree-based":
    forest = RandomForestRegressor(n_estimators=int(num_of_trees), max_depth=int(depth_of_trees), random_state=0)
    forest.fit(X, y)
    importances = forest.feature_importances_
    indices = importances.argsort()[::-1]
    features_to_keep = list(X.columns[indices[:int(num_of_features_to_keep)]])
    print(df[features_to_keep + ['species']])
elif feature_reduction_method == "Principal Component Analysis":
    X = df.drop('species', axis=1)
    pca = PCA(n_components=num_of_features_to_keep)
    principal_components = pca.fit_transform(X)
    columns = ['principal_component_' + str(i) for i in range(1, num_of_features_to_keep+1)]
    principal_df = pd.DataFrame(data=principal_components, columns=columns)
    print(principal_df.join(df['species']).head())
