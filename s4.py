import json
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Load the JSON file
with open('algoparams_from_ui') as f:
    config = json.load(f)

# Create the appropriate model object based on the prediction type
if config["design_state_data"]["target"]["prediction_type"] == 'Classification':
    if config["design_state_data"]["algorithms"]["RandomForestClassifier"]["is_selected"]==True :
            model = RandomForestClassifier()
            print(model)
    elif  config["design_state_data"]["algorithms"]["GBTClassifier"]["is_selected"]==True :
        model = GradientBoostingClassifier()
        print(model)
        
    elif  config["design_state_data"]["algorithms"]["LogisticRegression"]["is_selected"]==True :
        model = LogisticRegression()
        print(model)
    elif  config["design_state_data"]["algorithms"]["DecisionTreeClassifier"]["is_selected"]==True :
        model = DecisionTreeClassifier()
        print(model)
    elif  config["design_state_data"]["algorithms"]["SVM"]["is_selected"]==True :
        model = SVC()
        print(model)
    elif config["design_state_data"]["algorithms"]["KNN"]["is_selected"]==True :
        model = KNeighborsClassifier()
        print(model)
    elif  config["design_state_data"]["algorithms"]["extra_random_trees"]["is_selected"]==True :
        model = ExtraTreesClassifier()
        print(model)
    elif  config["design_state_data"]["algorithms"]["xg_boost"]["is_selected"]==True :
        model = XGBClassifier()
        print(model)
    elif config["design_state_data"]["algorithms"]["neural_network"]["is_selected"]==True :
        model = MLPClassifier()
        print(model)
    else:
        raise ValueError('Invalid model specified')

elif config["design_state_data"]["target"]["prediction_type"] == 'Regression':
    if  config["design_state_data"]["algorithms"]["RandomForestRegressor"]["is_selected"]==True :
        model = RandomForestRegressor()
        print(model)
    elif  config["design_state_data"]["algorithms"]["GBTRegressor"]["is_selected"]==True :
        model = GradientBoostingRegressor()
        print(model)
    elif  config["design_state_data"]["algorithms"]["LinearRegression"]["is_selected"]==True :
        model = LinearRegression()
        print(model)
    elif config["design_state_data"]["algorithms"]["RidgeRegression"]["is_selected"]==True :
        model = Ridge()
        print(model)
    elif  config["design_state_data"]["algorithms"]["LassoRegression"]["is_selected"]==True :
        model = Lasso()
        print(model)
    elif  config["design_state_data"]["algorithms"]["ElasticNetRegression"]["is_selected"]==True :
        model = ElasticNet()
        print(model)
    elif  config["design_state_data"]["algorithms"]["DecisionTreeRegressor"]["is_selected"]==True :
        model = DecisionTreeRegressor()
        print(model)
    elif config["design_state_data"]["algorithms"]["SGD"]["is_selected"]==True :
        model = SGDRegressor()
        print(model)
    elif config["design_state_data"]["algorithms"]["KNN"]["is_selected"]==True :
        model = KNeighborsRegressor()
        print(model)
    elif  config["design_state_data"]["algorithms"]["extra_random_trees"]["is_selected"]==True :
        model = ExtraTreesRegressor()
        print(model)
    elif  config["design_state_data"]["algorithms"]["xg_boost"]["is_selected"]==True :
        model = XGBRegressor()
        print(model)
    elif config["design_state_data"]["algorithms"]["neural_network"]["is_selected"]==True :
        model = MLPRegressor()
        print(model)
    

else:
    raise ValueError('Invalid prediction type specified')




