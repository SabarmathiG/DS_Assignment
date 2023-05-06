import json

# load JSON data from file
with open("algoparams_from_ui") as f:
    data = json.load(f)

# extract target and type of regression
target = data["design_state_data"]["target"]["target"]
prediction_type = data["design_state_data"]["target"]["prediction_type"]
type = data["design_state_data"]["target"]["type"]
partitioning = data["design_state_data"]["target"]["partitioning"]

print('"target": {')
print(f'\t"prediction_type": "{prediction_type}",')
print(f'\t"target": "{target}",')
print(f'\t"type": "{type}",')
print(f'\t"partitioning": {partitioning}')
print('}')

