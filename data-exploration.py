import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../../../Data/all_data.csv')
correlation = pd.DataFrame(df.drop(['start_of_week'],axis=1).corr()['total_boardings'])
model_data = df[correlation.dropna().index.values].drop(['local_business_condition_index'], axis=1)
features = model_data.drop(['total_boardings'], axis=1)
features['runtime_hours'] = features['runtime_hours'].rolling(window=10,min_periods=1,center=True).mean()
features = features.fillna(method='backfill')

cor_matrix = model_data.corr().abs()
to_drop = []
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
for i, j in upper_tri.iterrows():
    for x in range(len(j)):
        if j[x] > 0.9:
            if cor_matrix[i]['total_boardings'] > cor_matrix[j.index[x]]['total_boardings']:
                if j.index[x] not in to_drop:
                    to_drop.append(j.index[x])
            else:
                if i not in to_drop:
                    to_drop.append(i)

# sns.heatmap(cor_matrix.drop(to_drop,axis=1),cmap='viridis',annot=True)
# plt.show()

# print(to_drop)
print(features.drop(to_drop,axis=1).info())