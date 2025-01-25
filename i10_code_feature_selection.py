import numpy as np
import pandas as pd
i10_dx_values_unique = np.load('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\I10_DX_count.npy', allow_pickle=True)
i10_dx_values_unique = i10_dx_values_unique.tolist()
i10_dx_unique_values_3digits = list(set([x[0][:3] for x in i10_dx_values_unique]))
i10_dx_unique_values_3digits_count = []
for unique_dx_value in i10_dx_unique_values_3digits:
    unique_dx_value_count = np.array([x[1] for x in i10_dx_values_unique if x[0][:3] == unique_dx_value],dtype=np.int32).sum()
    i10_dx_unique_values_3digits_count.append([unique_dx_value,unique_dx_value_count])
i10_dx_unique_values_3digits_count = sorted(i10_dx_unique_values_3digits_count, key=lambda x: x[1], reverse=True)
selected_i10_code = [x[0] for x in i10_dx_unique_values_3digits_count if x[1] > 5]

filtered_csv = pd.read_csv('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\MD_SIDC_2020_CORE_filtered.csv', header=0)
i10_dx_columns = [col for col in filtered_csv.columns if 'I10_DX' in col]

sample_i10_one_hot = np.zeros((filtered_csv.shape[0], len(selected_i10_code)), dtype=np.int32)
i10_filtered_csv = filtered_csv.loc[:,i10_dx_columns]
i10_filtered_ndx = filtered_csv.loc[:,['I10_NDX']].values
sample_i10_list = [ x.values[:int(i10_filtered_ndx[_])].tolist() for _, x in i10_filtered_csv.iterrows()]
# if the first 3 digits of I10_DXn is in selected_i10_code, then set the corresponding position in sample_i10_one_hot to 1
for i in range(len(sample_i10_list)):
    for j in range(len(sample_i10_list[i])):
        if sample_i10_list[i][j][:3] in selected_i10_code:
            sample_i10_one_hot[i][selected_i10_code.index(sample_i10_list[i][j][:3])] = 1

labels = filtered_csv['LOS'].values

#use RF to select features from sample_i10_one_hot, the first mode is for all samples
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(sample_i10_one_hot, labels)
print(rf.feature_importances_)
print(rf.feature_importances_.sum())
#get the top 20 important features from selected_i10_code
importance = rf.feature_importances_
indices = np.argsort(importance)[::-1]
indices = indices[:20]
print(indices)
print([selected_i10_code[i] for i in indices])
#draw the importance of top 20 features into a bar chart
import matplotlib.pyplot as plt
plt.bar(np.arange(20), [importance[i] for i in indices])
plt.xticks(range(0, 20), [selected_i10_code[i] for i in indices], rotation=45, fontsize=6)
plt.title('Top 20 important diseases in MD_2020_SIDC, RF selected with regression on all samples')
plt.savefig('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\I10_DX_top20_RF_selected.png')
plt.clf()

#now we consider sample stays longer than 3 days
selected_samples = filtered_csv[filtered_csv['LOS'] > 3]
labels_mid = labels[filtered_csv['LOS'] > 3]
sample_i10_one_hot_mid = sample_i10_one_hot[filtered_csv['LOS'] > 3]
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(sample_i10_one_hot_mid, labels_mid)
importance = rf.feature_importances_
indices = np.argsort(importance)[::-1]
indices = indices[:20]
print(indices)
print([selected_i10_code[i] for i in indices])
plt.bar(np.arange(20), [importance[i] for i in indices])
plt.xticks(range(0, 20), [selected_i10_code[i] for i in indices], rotation=45, fontsize=6)
plt.title('Top 20 important diseases in MD_2020_SIDC, RF selected with regression on los>3 samples')
plt.savefig('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\I10_DX_top20_RF_selected_los3.png')
plt.clf()

#now we consider sample stays longer than 7 days
selected_samples = filtered_csv[filtered_csv['LOS'] > 7]
labels_long = labels[filtered_csv['LOS'] > 7]
sample_i10_one_hot_long = sample_i10_one_hot[filtered_csv['LOS'] > 7]
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(sample_i10_one_hot_long, labels_long)
importance = rf.feature_importances_
indices = np.argsort(importance)[::-1]
indices = indices[:20]
print(indices)
print([selected_i10_code[i] for i in indices])
plt.bar(np.arange(20), [importance[i] for i in indices])
plt.xticks(range(0, 20), [selected_i10_code[i] for i in indices], rotation=45, fontsize=6)
plt.title('Top 20 important diseases in MD_2020_SIDC, RF selected with regression on los>7 samples')
plt.savefig('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\I10_DX_top20_RF_selected_los7.png')
plt.clf()

#draw tsne plot for the top 20 important features
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2, random_state=42)
# tsne.fit(sample_i10_one_hot_mid[:,indices])
# embedding = tsne.embedding_
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 8))
# plt.scatter(
#     x=embedding[:, 0], y=embedding[:, 1],
#     c=labels_mid/labels_mid.max(),
#     cmap='viridis',
#     alpha=0.2,
#     s=10
# )
# plt.title('TSNE Projection of Top 20 Important Features')
# plt.xlabel('TSNE-1')
# plt.ylabel('TSNE-2')
# plt.savefig('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\TSNE_top20_RF_selected_los3.png')
# plt.clf()

