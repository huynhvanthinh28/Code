import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing, metrics, mixture
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
sns.set()
df = pd.read_excel('ASTM2.xlsx')

data = df.iloc[:,np.r_[8:20,20,21,24,25,27]]
soil_groups = df.iloc[:,6]
data_1 = data.fillna(0, inplace=True)

scaler = preprocessing.MinMaxScaler()
features_normal = scaler.fit_transform(data)


gmm_features = BayesianGaussianMixture(n_components=15,n_init=10, random_state=4).fit(features_normal)
gmm_predict = gmm_features.predict(features_normal)

empty_column=np.full((len(soil_groups), 1), np.inf)
new_columns = ['SOIL_GROUP','GMM_PRED','GMM_GROUP']
new_values = [soil_groups, gmm_predict,empty_column]

result_mapping = dict(zip(new_columns, new_values))

labeled_features = data[:]
labeled_features = labeled_features.assign(**result_mapping)

label_list = labeled_features['GMM_PRED'].unique()
for group in label_list:
	group_name=labeled_features[labeled_features['GMM_PRED']==group]['SOIL_GROUP'].mode()[0]
	labeled_features.loc[labeled_features['GMM_PRED']==group,'GMM_GROUP']=group_name

labeled_features.to_csv( 'gmm_2.csv')

count = []
for i in range(len(labeled_features['GMM_GROUP'])):
    if labeled_features["GMM_GROUP"][i] == labeled_features["SOIL_GROUP"][i]:
        count.append(1)

len(count)

print('Mức độ chính xác của phương pháp GMM là: ' + str(len(count)/len(labeled_features['GMM_GROUP'])*100))