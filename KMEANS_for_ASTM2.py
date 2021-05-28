import pandas as pd
import numpy as np
#import os
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from sklearn import preprocessing, metrics, mixture
from sklearn.cluster import KMeans

sns.set()
df = pd.read_excel('ASTM2.xlsx')
data = df.iloc[:,np.r_[8:20,20,21,24,25,27]]
soil_groups = df.iloc[:,6]
data_1 = data.fillna(0, inplace=True)

scaler = preprocessing.MinMaxScaler()
features_normal = scaler.fit_transform(data)
model = KMeans()

visualizer = KElbowVisualizer(model, k=(2,20), timings=True)
visualizer.fit(features_normal)
visualizer.show()

kmeans = KMeans(n_clusters=visualizer.elbow_value_).fit(features_normal)
kmeans_predict = pd.DataFrame(kmeans.labels_).iloc[:,0]

empty_column = np.full((len(soil_groups),1), np.inf)
new_columns = ['SOIL_GROUP', 'KMEANS_PRED', 'KMEANS_GROUP']
new_values = [soil_groups, kmeans_predict, empty_column]

result_mapping = dict(zip(new_columns, new_values))
labeled_features = data[:]
labeled_features = labeled_features.assign(**result_mapping)

label_list = labeled_features['KMEANS_PRED'].unique()
for group in label_list:
	group_name=labeled_features[labeled_features['KMEANS_PRED']==group]['SOIL_GROUP'].mode()[0]
	labeled_features.loc[labeled_features['KMEANS_PRED']==group,'KMEANS_GROUP']=group_name

labeled_features.to_csv( 'kmean_2.csv')

count = []
for i in range(len(labeled_features['KMEANS_GROUP'])):
    if labeled_features["KMEANS_GROUP"][i] == labeled_features["SOIL_GROUP"][i]:
        count.append(1)

len(count)
print('Mức độ chính xác của phương pháp KMeans là: ' + str(len(count)/len(labeled_features['KMEANS_GROUP'])*100))

