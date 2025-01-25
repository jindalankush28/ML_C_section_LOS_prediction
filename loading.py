import pandas as pd
import numpy as np
reloading = False
if reloading:
    csv_path = 'E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\MD_SIDC_2020_CORE.csv'
    csv_file = pd.read_csv(csv_path, header=0)

    #drop nan in LOS
    csv_file =csv_file.dropna(subset=['LOS'])
    print('in total  ',len(csv_file),' patients')

    all_columns = [col for col in csv_file.columns]
    i10_dx_columns = [col for col in csv_file.columns if 'I10' in col]
    mask = csv_file[i10_dx_columns].isin(['10D00Z1','10D00Z0','10D00Z2']).any(axis=1) #'10E0XZZ'
    # TODO: probably we can make a comparison with normal delivery cases in LOS prediction
    # mask2 = csv_file['I10_DELIVERY']
    filtered_csv = csv_file[mask]
    print('collected ',len(filtered_csv),' patients')
    filtered_csv.to_csv('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\MD_SIDC_2020_CORE_filtered.csv')
else:
    filtered_csv = pd.read_csv('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\MD_SIDC_2020_CORE_filtered.csv', header=0)
    print('in total  ',len(filtered_csv),' patients')

#save the first 10 samples to a csv file
# filtered_csv.head(10).to_csv('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\MD_SIDC_2020_CORE_filtered_example.csv')

#remove cols with 'I10', 'PRDAY' and 'DXPOA' in name
# filtered_csv = filtered_csv[filtered_csv.columns.drop(list(filtered_csv.filter(regex='I10')))]
# filtered_csv = filtered_csv[filtered_csv.columns.drop(list(filtered_csv.filter(regex='PRDAY')))]
# filtered_csv = filtered_csv[filtered_csv.columns.drop(list(filtered_csv.filter(regex='DXPOA')))]


all_columns = [col for col in filtered_csv.columns]

#Used common features

# MARITALSTATUSUB04 : 
# I: single
# M: married
# D: divorced
# X: separated
# nan: 
# W: widowed

# I10_NPR : number of procedures

# I10_NDX : number of diagnosis

# READMIT: 0: no readmission, 1: readmission (30 days)

# ATYPE: 1: emergency, 2: urgent, 3: elective, 4: newborn, 5: trauma, 6: others

# RACE: 1: white, 2: black, 3: hispanic, 4: asian, 5: native, 6: other

# BWT: birth weight (gram)

# DISPUB04: discharge status:
# 01	outine discharge (home/self care)
# 02	Another acute care hospital for inpatient care
# 03	Skilled nursing facility (SNF)
# 04	Intermediate care facility (ICF)
# 05	Designated Cancer Center or Children's hospital
# 06	Home health
# 07	Left/discontinued care against medical advice
# 20	Expired
# 21	Court/law enforcement
# 43	Discharged/transferred to a federal healthcare facility
# 50	Hospice - home
# 51	Hospice - medical facility
# 61	Transfer w/in institution to Medicare swing bed
# 62	Discharged/transferred to rehab facility or hospital unit
# 63	Discharged/transferred to Medicare certified long-term care hospital
# 64	Discharged/transferred to nursing facility certified under Medicaid bet not Medicare
# 65	Discharged/transferred to psychiatric hospital or psychiatric distinct part unit of a hospital
# 70	Discharged/transferred to another type of healthcare institution not otherwise defined

# DaysICU: number of days in ICU

# HISPANIC: solve it later, looking at what other people did

# HOSPST : hospital state

# Homeless: 0: not homeless, 1: homeless

# I10_PROCTYPE: not included
# 0	No ICD-10-PCS or CPT procedures
# 1	At least one ICD-10-PCS procedure; no CPT/HCPCS procedures
# 2	At least one CPT/HCPCS procedure; no ICD-10-PCS procedures
# 3	At least one ICD-10-PCS procedure and at least one CPT/HCPCS procedure

# I10_SERVICELINE:
# 1: Maternal and neonatal
# 2: Mental health/substance use
# 3: Injury
# 4: Surgical
# 5: Medical 

# PAY1~PAY3 Expected primary payer, uniform
# 1 Medicare
# 2 Medicaid
# 3 Private insurance
# 4 Self-pay
# 5 No charge
# 6 Other

# OS_TIME： observation time in hours

# PL_NCHS: Patient Location: NCHS Urban-Rural Code
# 1: "Central" counties of metro areas of >=1 million population
# 2: "Fringe" counties of metro areas of >=1 million population
# 3: Counties in metro areas of 250,000-999,999 population
# 4: Counties in metro areas of 50,000-249,999 population
# 5: Micropolitan counties
# 6: Not metropolitan or micropolitan counties

# PL_RUCC Patient location: Rural-Urban Continuum (RUCC) Codes
# 1 Metropolitan areas of 1 million population or more
# 2 Metropolitan areas of 250,000 to 1 million population
# 3 Metropolitan areas of fewer than 250,000 population
# 4 Urban population of 20,000 or more, adjacent to a metropolitan area
# 5 Urban population of 20,000 or more, not adjacent to a metropolitan area
# 6 Urban population of 2,500 to 19,999, adjacent to a metropolitan area
# 7 Urban population of 2,500 to 19,999, not adjacent to a metropolitan area
# 8 Completely rural or less than 2,500 urban population, adjacent to a metropolitan area
# 9 Completely rural or less than 2,500 urban population, not adjacent to a metropolitan area

# PL_UIC Patient location: Urban influence codes
# 1 Metro - Large metro area of 1 million residents or more
# 2 Metro - Small metro area of less than 1 million residents
# 3 Non-Metro - Micropolitan adjacent to large metro area
# 4 Non-Metro - Noncore adjacent to large metro area
# 5 Non-Metro - Micropolitan adjacent to small metro area
# 6 Non-Metro - Noncore adjacent to small metro area and contains a town of at least 2,500 residents
# 7 Non-Metro - Noncore adjacent to small metro area and does not contain a town of at least 2,500 residents
# 8 Non-Metro - Micropolitan not adjacent to a metro area
# 9 Non-Metro - Noncore adjacent to micro area and contains a town of at least 2,500 residents
# 10 Non-Metro - Noncore adjacent to micro area and does not contain a town of at least 2,500 residents
# 11 Non-Metro - Noncore not adjacent to metro or micro area and contains a town of at least 2,500 residents
# 12 Non-Metro - Noncore not adjacent to metro or micro area and does not contain a town of at least 2,500 residents

# PrimLang: primary language categories of language

common_features = ['ZIP','RACE','AGE','FEMALE','ATYPE','READMIT','I10_NDX','I10_NPR','MARITALSTATUSUB04','BWT','DISPUB04','Homeless',
                   'I10_SERVICELINE','PAY1','PAY2','PAY3','PL_NCHS']

# exlude sample with 'DIED' == 1
# double check the I10_DELIVERY column

# # make statistics on features in cols with I10_PRn unque values
# i10_pr_columns = [col for col in filtered_csv.columns if 'I10_PR' in col]
# # exclude element I10_PROCTYPE
# i10_pr_columns.remove('I10_PROCTYPE')

# # get unique values in all cols in i10_pr_columns
# i10_pr_values_unique = []
# for col in i10_pr_columns:
#     i10_pr_values_unique.extend(filtered_csv[col].dropna(axis=0, how='all').unique())
# i10_pr_values_unique = list(set(i10_pr_values_unique))

# #make statistics of top-k mode values in i10_pr_values_unique
# i10_pr_values_unique_count = [[value,filtered_csv[i10_pr_columns].isin([value]).any(axis=1).sum()] for value in i10_pr_values_unique]
# i10_pr_values_unique_count = sorted(i10_pr_values_unique_count, key=lambda x: x[1], reverse=True)
# import matplotlib.pyplot as plt
# plt.bar(np.arange(1,21), [i10_pr_values_unique_count[i][1] for i in range(1, 21)])
# plt.xticks(range(0, 20), [i10_pr_values_unique_count[i][0] for i in range(1, 21)], rotation=45, fontsize=6)
# plt.title('Top 20 diseases in MD_2020_SIDC')
# plt.savefig('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\I10_PR_top20.png')
# plt.clf()

# make statistics on features in cols with I10_PRn unque values
i10_dx_columns = [col for col in filtered_csv.columns if 'I10_DX' in col]

# get unique values in all cols in i10_dx_columns
i10_dx_values_unique = []
for col in i10_dx_columns:
    i10_dx_values_unique.extend(filtered_csv[col].dropna(axis=0, how='all').unique())
i10_dx_values_unique = list(set(i10_dx_values_unique))

#make statistics of top-k mode values in i10_dx_values_unique
# for unique_dx_value in i10_dx_values_unique:
#     i10_dx_values_unique_count.append([unique_dx_value,filtered_csv[i10_dx_columns].isin([unique_dx_value]).any(axis=1).sum()])
i10_dx_values_unique_count = [[value,filtered_csv[i10_dx_columns].isin([value]).any(axis=1).sum()] for value in i10_dx_values_unique]
i10_dx_values_unique_count = sorted(i10_dx_values_unique_count, key=lambda x: x[1], reverse=True)
np.save('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\I10_DX_count.npy', i10_dx_values_unique_count, allow_pickle=True)
import matplotlib.pyplot as plt
plt.bar(np.arange(1,21), [i10_dx_values_unique_count[i][1] for i in range(1, 21)])
plt.xticks(range(0, 20), [i10_dx_values_unique_count[i][0] for i in range(1, 21)], rotation=45, fontsize=6)
plt.title('Top 20 diseases in MD_2020_SIDC')
plt.savefig('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\I10_DX_top20.png')
plt.clf()

#merge the count by first 3 digits
i10_dx_unique_values_3digits = list(set([x[:3] for x in i10_dx_values_unique]))
i10_dx_unique_values_3digits_count = []
for unique_dx_value in i10_dx_unique_values_3digits:
    unique_dx_value_count = np.array([x[1] for x in i10_dx_values_unique_count if x[0][:3] == unique_dx_value]).sum()
    i10_dx_unique_values_3digits_count.append([unique_dx_value,unique_dx_value_count])
i10_dx_unique_values_3digits_count = sorted(i10_dx_unique_values_3digits_count, key=lambda x: x[1], reverse=True)
plt.bar(np.arange(1,21), [i10_dx_unique_values_3digits_count[i][1] for i in range(1, 21)])
plt.xticks(range(0, 20), [i10_dx_unique_values_3digits_count[i][0] for i in range(1, 21)], rotation=45, fontsize=6)
plt.title('Top 20 diseases in MD_2020_SIDC')
plt.savefig('E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\I10_DX_top20_3d.png')
plt.clf()





 