import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing, metrics, mixture
from sklearn.mixture import GaussianMixture


sns.set()
df = pd.read_excel('ASTM1.xlsx')
print(df)
data = df.iloc[:,np.r_[8:19,19,20,22,23,24,26]]
soil_groups = df.iloc[:,6]
depth = df['DEPTH1']
data_1 = data.fillna(0, inplace=True)

scaler = preprocessing.MinMaxScaler()
features_normal = scaler.fit_transform(data)

n_components = np.arange(1,20)

models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(features_normal) for n in n_components]


plt.plot(n_components, [m.aic(features_normal) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');
plt.show()


gmm_features = GaussianMixture(n_components=9).fit(features_normal)
gmm_predict = gmm_features.predict(features_normal)

empty_column=np.full((len(soil_groups), 1), np.inf)
new_columns = ['DEPTH','SOIL_GROUP','GMM_PRED','GMM_GROUP']
new_values = [depth,soil_groups, gmm_predict,empty_column]

result_mapping = dict(zip(new_columns, new_values))

labeled_features = data[:]
labeled_features = labeled_features.assign(**result_mapping)

label_list = labeled_features['GMM_PRED'].unique()
for group in label_list:
	group_name=labeled_features[labeled_features['GMM_PRED']==group]['SOIL_GROUP'].mode()[0]
	labeled_features.loc[labeled_features['GMM_PRED']==group,'GMM_GROUP']=group_name

labeled_features.to_csv( 'gmm_1.csv')

count = []
for i in range(len(labeled_features['GMM_GROUP'])):
    if labeled_features["GMM_GROUP"][i] == labeled_features["SOIL_GROUP"][i]:
        count.append(1)

len(count)

print('Mức độ chính xác của phương pháp GMM là: ' + str(len(count)/len(labeled_features['GMM_GROUP'])*100))