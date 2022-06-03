import math
import pandas as pd
import numpy as np
import random


def calEntropy(dataset):
    dataset = np.array(dataset)
    data_len = len(dataset)
    # LabelCount: number of samples for each label category
    labelCount = {}

    for row in dataset:
        label = row[-1]
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1

    result = 0
    for key in labelCount.keys():
        prob = labelCount[key]/data_len # Frequency
        result -= prob*math.log2(prob)
    return result


# Choose the best feature to split based on the information gain criteria
def BestFeatureToSplit(dataset:pd.DataFrame):
    features = dataset.columns.values[0:-1]
    best_feat = ''
    max_Gain = 0
    for feat in features:
      # Entropy before split
      entropy = calEntropy(dataset)
      # Records the information gain of "feat"
      feat_values = dataset[feat].unique()
      # Initialize it with the max IG it can reach, and do subtraction from it in the loop
      feat_Gain = entropy

      for val in feat_values:
        # data_v is the sub dataset when feat==val
        data_v = dataset[dataset[feat]==val]
        # ratio_v is the ratio of data_v to dataset
        ratio_v = (data_v.shape[0])/(dataset.shape[0])
        tmp_entropy = calEntropy(data_v)
        feat_Gain -= ratio_v*tmp_entropy

      # Print('gain: ',feat_Gain,'\n')
      if feat_Gain > max_Gain:
        max_Gain = feat_Gain
        best_feat = feat
    return max_Gain, best_feat


# Split the dataset based on the feature chosen in BestFeatureToSplit
def splitData(dataset, feat, value):
    feat_values = dataset.columns.values
    ret_feats = [f for f in feat_values if f != feat]
    # Remove the used features
    ret_dataset = dataset.loc[dataset[feat]==value, ret_feats]
    return ret_dataset


def countMajority(labelList):
    '''
    labelList: only the label column of the dataset
    '''
    data = pd.Series(labelList)
    compValue = data.value_counts()[0] # gives the label with maximun number of samples
    newDataList = [] # to store labels with same amount of samples
    newDataList.append(data.value_counts().index[0])
    for i in range(1,len(data.value_counts())):
      if compValue == data.value_counts()[i]:
        newDataList.append(data.value_counts().index[i])
    return random.choice(newDataList) # choose one label from the newDataList randomly


def buildDecisionTree(ori_data, dataset):#ori_data: the original dataset with everything in it
    labelList = list(dataset['Enjoy']) 

    # when samples in dataset all have the same label, stop recursion
    if labelList.count(labelList[0]) == len(labelList):
      return labelList[0]

    # all features are used up, no more features can be used to split on
    if len(dataset.columns.values) == 1:
      return countMajority(labelList)

    # find out the best feature to split on
    _, bestFeat = BestFeatureToSplit(dataset)

    # if there are contradictory samples (having same value on all features,
    # but have different labels), then follow the majority rule
    if bestFeat == '':
      return countMajority(labelList)

    # As we split the data, the sub dataset may not cover all the values of the feature.
    # So here ori_data must be the original dataset which haven't been split
    feat_values = ori_data[bestFeat].unique()

    decisionTree = {bestFeat:{}}  # add a branch

    # split the dataset
    for val in feat_values:
      sub_dataset = splitData(dataset, bestFeat, val)
      if len(sub_dataset) == 0:
        decisionTree[bestFeat][val] = countMajority(labelList)
      else:
        decisionTree[bestFeat][val] = buildDecisionTree(ori_data, sub_dataset)

    return decisionTree


# print the tree
def print_tree(tree, level=0):
  if not isinstance(tree, dict):
    print('\t'*level+'-result: '+tree)
    return
  for feature, subtree in tree.items():
    for value in subtree.keys():
      print('\t'*level+'-'+feature+': '+value)
      branch = subtree[value]
      print_tree(branch, level+1)
  return


# Load data
df_enjoy = pd.read_csv('dt_data.txt', sep=", ", header = None)
df_enjoy.columns = ["Occupied", "Price", "Music", "Location", "VIP", "Favorite Beer", "Enjoy"]

# Training
myTree = buildDecisionTree(df_enjoy, df_enjoy)

# Print
print_tree(myTree)

# Prediction
# predict values:
test_data = {'Occupied': 'Moderate', 'Price': 'Cheap', 'Music': 'Loud', 'Location': 'City-Center', 'VIP': 'No', 'Favorite Beer': 'No'}

key = next(iter(myTree))
result = myTree
while isinstance(result, dict):
  val = test_data[key]
  result = result[key][val]
  key = next(iter(result))

print(result)