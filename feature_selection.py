import numpy as np
i10_dx_values_unique = np.load('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\I10_DX_count.npy', allow_pickle=True)
i10_dx_values_unique = i10_dx_values_unique.tolist()
I10_dx_unique_values_3digits = list(set([x[0][:3] for x in i10_dx_values_unique]))
i10_dx_unique_values_3digits_count = []
for unique_dx_value in i10_dx_unique_values_3digits:
    unique_dx_value_count = np.array([x[1] for x in i10_dx_values_unique_count if x[0][:3] == unique_dx_value]).sum()
    i10_dx_unique_values_3digits_count.append([unique_dx_value,unique_dx_value_count])
i10_dx_unique_values_3digits_count = sorted(i10_dx_unique_values_3digits_count, key=lambda x: x[1], reverse=True)