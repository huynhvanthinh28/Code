import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing, metrics, mixture
from sklearn.cluster import MeanShift, estimate_bandwidth

sns.set()
df = pd.read_excel('TCVN.xlsx')

data = df.iloc[:,np.r_[7:19,20,19,23,24,26]]
soil_groups = df.iloc[:,5]
depth = df['DEPTH1']
data_1 = data.fillna(0, inplace=True)


scaler = preprocessing.MinMaxScaler()
features_normal = scaler.fit_transform(data)

est_bandwidth = estimate_bandwidth(features_normal, quantile=.08,
n_samples=500)

ms = MeanShift(bandwidth=est_bandwidth,bin_seeding=True).fit(features_normal)
labels = ms.labels_
meanshift_predict = pd.DataFrame(ms.labels_).iloc[:,0]

empty_column = np.full((len(soil_groups),1), np.inf)
new_columns = ['DEPTH','SOIL_GROUP', 'MEANSHIFT_PRED', 'MEANSHIFT_GROUP']
new_values = [depth,soil_groups, meanshift_predict, empty_column]
result_mapping = dict(zip(new_columns, new_values))
labeled_features = data[:]
labeled_features = labeled_features.assign(**result_mapping)
label_list = labeled_features['MEANSHIFT_PRED'].unique()
for group in label_list:
	group_name=labeled_features[labeled_features['MEANSHIFT_PRED']==group]['SOIL_GROUP'].mode()[0]
	labeled_features.loc[labeled_features['MEANSHIFT_PRED']==group,'MEANSHIFT_GROUP']=group_name

labeled_features.to_csv('meanshift_3.csv')

count = []
for i in range(len(labeled_features['MEANSHIFT_GROUP'])):
    if labeled_features["MEANSHIFT_GROUP"][i] == labeled_features["SOIL_GROUP"][i]:
        count.append(1)
print('Mức độ chính xác của phương pháp Meanshift là: ' + str(len(count)/len(labeled_features['MEANSHIFT_GROUP'])*100))
