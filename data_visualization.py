import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
filtered_csv = pd.read_csv('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\MD_SIDC_2020_CORE_filtered.csv', header=0)

# data analysis for the features
common_features = ['ZIP','RACE','AGE','FEMALE','ATYPE','READMIT','I10_NDX','I10_NPR','MARITALSTATUSUB04','BWT','DISPUB04','Homeless',
                   'I10_SERVICELINE','PAY1','PAY2','PAY3','PL_NCHS']
LOS = filtered_csv['LOS']
filtered_csv = filtered_csv[common_features]

specific_csv = pd.merge(filtered_csv, LOS, left_index=True, right_index=True)
#extract the first 3 digits of zip code
specific_csv['ZIPFIRST3'] = specific_csv['ZIP'].apply(lambda x: str(x)[:3])
#transform AGE to float
specific_csv['AGE'] = specific_csv['AGE'].apply(lambda x: 0 if not str(x).isdigit() else x)
specific_csv['AGE'] = specific_csv['AGE'].astype(float)
result_dict = {}
for selected_variable in ['RACE','ATYPE','READMIT','MARITALSTATUSUB04','DISPUB04','Homeless','I10_SERVICELINE','PAY1','PAY2','PAY3','PL_NCHS','ZIPFIRST3']:
    model = ols('LOS ~ C('+selected_variable+')', data=specific_csv).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    eta_squared = anova_table['sum_sq']['C('+selected_variable+')'] / anova_table['sum_sq'].sum()
    print(selected_variable, eta_squared)
    result_dict[selected_variable] = eta_squared
    
for selected_variable in ['AGE','I10_NDX','I10_NPR','BWT']:
    model = ols('LOS ~ '+selected_variable, data=specific_csv).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    eta_squared = anova_table['sum_sq'][selected_variable] / anova_table['sum_sq'].sum()
    print(selected_variable, eta_squared)
    result_dict[selected_variable] = eta_squared


import umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), ['I10_NDX','I10_NPR']),
    (OneHotEncoder(), ['READMIT','DISPUB04','ZIPFIRST3']),
)
# preprocessor = make_column_transformer(
#     (StandardScaler(), ['I10_NDX','I10_NPR','AGE']),
#     (OneHotEncoder(), ['RACE','ATYPE','READMIT','MARITALSTATUSUB04','DISPUB04','Homeless','I10_SERVICELINE','PAY1','PL_NCHS','ZIPFIRST3']),
# )
umap_model = umap.UMAP(n_neighbors=50, min_dist=0.1,spread=2,verbose=True,n_epochs=500)
pipeline = make_pipeline(preprocessor, umap_model)

#drop all samples contains nan removal PAY2 and PAY3, BWT cols
used_samples = specific_csv[[x for x in specific_csv.columns if x not in ['PAY2','PAY3','BWT','ZIP']]]
used_samples = used_samples.dropna(axis=0, how='any')
#smpling based on the value of los
used_samples = used_samples.sort_values(by='LOS')
used_samples = used_samples.iloc[10000:20000]
used_los = used_samples['LOS']
embedding = pipeline.fit_transform(used_samples)

# Plotting
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=embedding[:, 0], y=embedding[:, 1],
    hue=used_los/used_los.max(),
    palette='viridis',
    alpha=0.2,
    s=10
)
plt.title('UMAP Projection of Selected Features')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.savefig('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\UMAP.png')
# plt.savefig('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\UMAP_all.png')
plt.clf()

from sklearn.ensemble import RandomForestRegressor  
from sklearn.datasets import make_classification
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), ['I10_NDX','I10_NPR','AGE']),
    (OneHotEncoder(), ['RACE','ATYPE','READMIT','MARITALSTATUSUB04','DISPUB04','Homeless','I10_SERVICELINE','PAY1','PL_NCHS','ZIPFIRST3']),
)
model = make_pipeline(preprocessor,RandomForestRegressor(random_state=42))
#drop all samples contains nan removal PAY2 and PAY3, BWT cols
used_samples = specific_csv[[x for x in specific_csv.columns if x not in ['PAY2','PAY3','BWT','ZIP']]]
used_samples = used_samples.dropna(axis=0, how='any')
model.fit(used_samples[[x for x in used_samples.columns if x not in ['LOS']]],used_samples['LOS'])
importances = model.named_steps['randomforestregressor'].feature_importances_
feature_names = model.named_steps['columntransformer'].transformers_[1][1].get_feature_names_out(['RACE','ATYPE','READMIT','MARITALSTATUSUB04','DISPUB04','Homeless','I10_SERVICELINE','PAY1','PL_NCHS','ZIPFIRST3'])
feature_names = np.concatenate((['I10_NDX','I10_NPR','AGE'], feature_names))

#get prediction performance 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

mse = mean_squared_error(used_samples['LOS'], model.predict(used_samples[[x for x in used_samples.columns if x not in ['LOS']]]))
r2 = r2_score(used_samples['LOS'], model.predict(used_samples[[x for x in used_samples.columns if x not in ['LOS']]]))
#plotting a bar plot for feature importance
import matplotlib.pyplot as plt
plt.bar(np.arange(1,21), importances[np.argsort(importances)[-20:]])
plt.xticks(range(0, 20), feature_names[np.argsort(importances)[-20:]], rotation=45, fontsize=6)
plt.title('Top 20 features selected by RF in MD_2020_SIDC')
plt.savefig('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\feature_importance_RF.png')
plt.clf()


