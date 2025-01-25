import numpy as np
import pandas as pd
# train_val_result = np.load('/home/local/zding20/exp2_v2/temp/LoS_Benchmark/output/MLP/randomized_search/train_val_result.npy',allow_pickle=True)
# a = train_val_result[0,:,1,:]
# np.mean(a,axis=0)
filtered_csv = pd.read_csv('/home/local/zding20/exp2_v2/temp/LoS_Benchmark/data/combined_2016to2019_all.csv')
filtered_csv = filtered_csv.head(10)
filtered_csv.to_csv('/home/local/zding20/exp2_v2/temp/LoS_Benchmark/data/combined_2016to2019_all_head.csv',index=False)