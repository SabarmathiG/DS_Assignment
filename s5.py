from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=FutureWarning)

with open('algoparams_from_ui', 'r') as f:
    data = json.load(f)

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

model_config = data["design_state_data"]["algorithms"]

#  a dictionary of models with their  hyperparameters
models = {
        
    "RandomForestClassifier": {"model": RandomForestClassifier(), "params": {'n_estimators': range(model_config["RandomForestClassifier"]['min_trees'], model_config["RandomForestClassifier"]['max_trees']+1),
    'max_depth': range(model_config["RandomForestClassifier"]['min_depth'], model_config["RandomForestClassifier"]['max_depth']+1),
    'min_samples_leaf': range(model_config["RandomForestClassifier"]['min_samples_per_leaf_min_value'], model_config["RandomForestClassifier"]['min_samples_per_leaf_max_value']+1)}},

    "RandomForestRegressor": {"model": RandomForestRegressor(), "params": {    'n_estimators': range(model_config["RandomForestRegressor"]['min_trees'], model_config["RandomForestRegressor"]['max_trees']+1),
    'max_depth': range(model_config["RandomForestRegressor"]['min_depth'], model_config["RandomForestRegressor"]['max_depth']+1),
    'min_samples_leaf': range(model_config["RandomForestRegressor"]['min_samples_per_leaf_min_value'], model_config["RandomForestRegressor"]['min_samples_per_leaf_max_value']+1)
}},




 "GBTClassifier": {"model": GradientBoostingClassifier(), "params": {'n_estimators': model_config["GBTClassifier"]['num_of_BoostingStages'],
'learning_rate': [0.1] ,
    'subsample': np.arange(model_config["GBTClassifier"]['min_subsample'], model_config["GBTClassifier"]['max_subsample']+0.1, 0.1),
    'max_depth': range(model_config["GBTClassifier"]['min_depth'], model_config["GBTClassifier"]['max_depth']+1),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}},
        "GBTRegressor": {"model": GradientBoostingRegressor(), "params": {'n_estimators': model_config["GBTRegressor"]['num_of_BoostingStages'],
                                                                          'learning_rate': [0.1] ,
    'subsample': np.arange(model_config["GBTRegressor"]['min_subsample'], model_config["GBTRegressor"]['max_subsample']+0.1, 0.1),
    'max_depth': range(model_config["GBTRegressor"]['min_depth'], model_config["GBTRegressor"]['max_depth']+1),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}},

          "LinearRegression": {"model": LinearRegression(), "params": {}},

 "LogisticRegression": {"model": LogisticRegression(), "params": {   
    'C': np.linspace(model_config["LogisticRegression"]['min_regparam'], model_config["LogisticRegression"]['max_regparam'], 10),
    'solver': ['liblinear'],
    'l1_ratio': np.linspace(model_config["LogisticRegression"]['min_elasticnet'], model_config["LogisticRegression"]['max_elasticnet'], 10),
    'max_iter': range(model_config["LogisticRegression"]['min_iter'], model_config["LogisticRegression"]['max_iter']+1)
}},




     "RidgeRegression": {"model": Ridge(), "params": {  'alpha': [0.1, 1.0, 10.0],
        'fit_intercept': [True, False],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'max_iter': [None, 100, 1000],
        'tol': [0.001, 0.0001, 0.00001],
        'random_state': [42]
}},



          "LassoRegression": {"model": Lasso(), "params": { 'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                                 'max_iter': range(model_config['LassoRegression']['min_iter'], model_config['LassoRegression']['max_iter']+1),
                                                 'tol': [0.0001, 0.001, 0.01]}},


  "ElasticNetRegression": {"model": ElasticNet(), "params": { 'alpha': [0.1, 0.5, 1.0],
    'l1_ratio': np.arange(model_config['ElasticNetRegression']['min_elasticnet'], model_config['ElasticNetRegression']['max_elasticnet']+0.1, 0.1),
    'max_iter': range(model_config['ElasticNetRegression']['min_iter'], model_config['ElasticNetRegression']['max_iter']+1)
}},





          "xg_boost": {"model": XGBRegressor(), 'params': {    'n_estimators': range(50, 101),

           'learning_rate': model_config['xg_boost']['learningRate'],
           'max_depth': model_config['xg_boost']['max_depth_of_tree'],
           'reg_alpha': model_config['xg_boost']['l1_regularization'],
           'reg_lambda': model_config['xg_boost']['l2_regularization'],
           'gamma': model_config['xg_boost']['gamma'],
           'min_child_weight': model_config['xg_boost']['min_child_weight'],
           'subsample': [s/100.0 for s in model_config['xg_boost']['sub_sample']],
           'colsample_bytree': [c/100.0 for c in model_config['xg_boost']['col_sample_by_tree']]
          }
},




 "DecisionTreeRegressor": {
    "model": DecisionTreeRegressor(),
    "params": {
        'max_depth': range(model_config['DecisionTreeRegressor']['min_depth'], model_config['DecisionTreeRegressor']['max_depth']+1),
        'criterion': ['mse', 'friedman_mse'] if not model_config['DecisionTreeRegressor']['use_entropy'] else ['friedman_mse'],
        'min_samples_leaf': model_config['DecisionTreeRegressor']['min_samples_per_leaf'],
        'splitter': ['best'] if model_config['DecisionTreeRegressor']['use_best'] else ['random']
    }
},



        
"DecisionTreeClassifier": {"model": DecisionTreeClassifier(), "params": { 'criterion': ['gini', 'entropy'] if model_config['DecisionTreeClassifier']['use_gini'] or model_config['DecisionTreeClassifier']['use_entropy'] else ['gini'],
    'max_depth': range(model_config['DecisionTreeClassifier']['min_depth'], model_config["DecisionTreeClassifier"]['max_depth']+1),
    'min_samples_leaf': model_config['DecisionTreeClassifier']['min_samples_per_leaf']

                 }},

"SVM": {"model": SVC(), "params": {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                                   'C': model_config["SVM"]['c_value'],
                                   'gamma': [0.1, 0.01, 0.001],
                                   'tol': [10**(-model_config["SVM"]['tolerance'])],
                                   'max_iter': [10**model_config["SVM"]['max_iterations']]}}
,


         "SGD": {"model": SGDRegressor(), "params": {  
    'max_iter': [model_config["SGD"]['max_iterations']] if model_config["SGD"]['max_iterations'] else [1000],
    'tol': [model_config["SGD"]['tolerance']],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'alpha': model_config["SGD"]['alpha_value'],
    'l1_ratio': [0.15, 0.25, 0.5, 0.75]
}},


"extra_random_trees": {"model": ExtraTreesRegressor(), "params": { 'n_estimators': model_config["extra_random_trees"]['num_of_trees'],
'max_features': [0.5],
'max_depth': model_config["extra_random_trees"]['max_depth'],
'min_samples_leaf': model_config["extra_random_trees"]['min_samples_per_leaf']}},


    "neural_network": {"model": MLPRegressor(), "params": {    
    'hidden_layer_sizes': model_config["neural_network"]['hidden_layer_sizes'],
    'activation': ['tanh'] if model_config["neural_network"]['activation'] == '' else model_config["neural_network"]['activation'],
    'alpha': [model_config["neural_network"]['alpha_value']],
    'max_iter': [1000],
    'tol': [model_config["neural_network"]['convergence_tolerance']],
    'early_stopping': [model_config["neural_network"]['early_stopping']],
    'solver': [model_config["neural_network"]['solver'].lower()],
    'shuffle': [model_config["neural_network"]['shuffle_data']],
    'learning_rate_init': [0.1] if model_config["neural_network"]['initial_learning_rate'] == 0 else model_config["neural_network"]['initial_learning_rate'],

    'epsilon': [0.1] if model_config["neural_network"]['epsilon'] == 0 else model_config["neural_network"]['epsilon'],

    'batch_size': ['auto'] if model_config["neural_network"]['automatic_batching'] else [model_config["neural_network"]['batch_size']],
    'beta_1': [model_config["neural_network"]['beta_1']],
    'beta_2': [model_config["neural_network"]['beta_2']],
    'power_t': [model_config["neural_network"]['power_t']],
    'momentum': [model_config["neural_network"]['momentum']],
    'nesterovs_momentum': [model_config["neural_network"]['use_nesterov_momentum']]
}}
    }


for model_name, model_params in models.items():
    print("Running {}...".format(model_name))
    model = model_params["model"]
    params = model_params["params"]
    clf = GridSearchCV(model, params, cv=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)