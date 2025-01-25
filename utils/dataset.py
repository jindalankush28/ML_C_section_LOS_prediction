import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from hcuppy.elixhauser import ElixhauserEngine
ee=ElixhauserEngine()
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

def simple_loading_process(args):
    '''example of loading process'''

    if args.dataset_regenerate:
        print('regenerating the cohert from the all data')
        csv_path = args.all_data_dir
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
        filtered_csv.to_csv(args.cohert_dir)

    else:
        print('loading the cohert from the filtered data')
        filtered_csv = pd.read_csv(args.cohert_dir, header=0)
        print('in total  ',len(filtered_csv),' patients')
        
    #extract the label
    label = filtered_csv['LOS']
    if args.normalization:
    #min-max normalization
        label = label / 100
    
    common_features = ['ZIP','RACE','AGE','FEMALE','ATYPE','READMIT','I10_NDX','I10_NPR','MARITALSTATUSUB04','BWT','DISPUB04','Homeless',
                   'I10_SERVICELINE','PAY1','PAY2','PAY3','PL_NCHS']
    filtered_csv = filtered_csv[common_features]
    
    data = []
    #extract the first 3 digits of zip code
    # filtered_csv['ZIPFIRST3'] = filtered_csv['ZIP'].apply(lambda x: str(x)[:3])
    #encoding ZIPFIRST3 into one-hot
    # data.append(pd.get_dummies(filtered_csv['ZIPFIRST3'], prefix='ZIPFIRST3'))
    #encoding RACE, FEMALE, ATYPE, READMIT, MARITALSTATUSUB04, DISPUB04, Homeless, I10_SERVICELINE, PAY1, PL_NCHS into one-hot
    for feature in ['RACE', 'FEMALE', 'ATYPE', 'READMIT', 'MARITALSTATUSUB04', 'DISPUB04', 'Homeless', 'I10_SERVICELINE', 'PAY1', 'PL_NCHS']:
        data.append(pd.get_dummies(filtered_csv[feature], prefix=feature))
    #encoding AGE, I10_NDX, I10_NPR, by normalization
    for feature in ['AGE', 'I10_NDX', 'I10_NPR']:
        #assign the nondigit value to 0
        filtered_csv[feature] = filtered_csv[feature].apply(lambda x: 0 if not str(x).isdigit() else x)
        filtered_csv[feature] = filtered_csv[feature].astype(float)
        data.append((filtered_csv[feature]-filtered_csv[feature].mean())/filtered_csv[feature].std())
    
    #computing the mean and var for nonempty BWT and normalize the nonempty BWT, then assign 0 to empty BWT
    filtered_csv['BWT'] = (filtered_csv['BWT']-filtered_csv['BWT'][~filtered_csv['BWT'].isna()].mean())/filtered_csv['BWT'][~filtered_csv['BWT'].isna()].std()
    filtered_csv['BWT'][filtered_csv['BWT'].isna()] = 0
    
    
    #encoding if PAY2, PAY3 is nonempty by 1, else 0
    for feature in ['PAY2', 'PAY3']:
        data.append(filtered_csv[feature].notnull())
    
    data = pd.concat(data, axis=1)

    return data, label

def tvt_split(args,data,label):
    '''split the dataset into train, val, test'''
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=args.seed)
    train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.2, random_state=args.seed)
    return train_data, val_data, test_data, train_label, val_label, test_label

def tt_split(args,data,label):
    '''split the dataset into train, test'''
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=args.seed)
    return train_data, test_data, train_label, test_label

def combined_data_process(args):
    csv_path = args.all_data_dir
    csv_file = pd.read_csv(csv_path, header=0)
    filtered_csv =csv_file.dropna(subset=['LOS'])
    print('in total  ',len(filtered_csv),' patients')
    all_columns = [col for col in csv_file.columns]
    #extract the label
    label = filtered_csv['LOS']
    if args.normalization:
    #min-max normalization
        label = label / 10
    
    common_features = ['ZIP','RACE','AGE','FEMALE','ATYPE','READMIT','I10_NDX','I10_NPR','MARITALSTATUSUB04','BWT','DISPUB04','Homeless',
                   'I10_SERVICELINE','PAY1','PAY2','PAY3','PL_NCHS','YEAR']
    filtered_csv = filtered_csv[common_features]
    data = []
    #extract the first 3 digits of zip code
    # filtered_csv['ZIPFIRST3'] = filtered_csv['ZIP'].apply(lambda x: str(x)[:3])
    #encoding ZIPFIRST3 into one-hot
    # data.append(pd.get_dummies(filtered_csv['ZIPFIRST3'], prefix='ZIPFIRST3'))
    #encoding RACE, FEMALE, ATYPE, READMIT, MARITALSTATUSUB04, DISPUB04, Homeless, I10_SERVICELINE, PAY1, PL_NCHS into one-hot
    # import matplotlib.pyplot as plt
    # plt.scatter(label,filtered_csv['AGE'])
    # plt.savefig('/home/local/zding20/exp2_v2/temp/LoS_Benchmark/output/MLP/randomized_search/age_los.png')
    # plt.clf()
    
    # plt.hist(label,bins = 100)
    # #log scale on y axis
    # plt.yscale('log')
    # plt.savefig('/home/local/zding20/exp2_v2/temp/LoS_Benchmark/output/MLP/randomized_search/los.png')
    # plt.clf()
    
    for feature in ['RACE', 'FEMALE', 'ATYPE', 'READMIT', 'MARITALSTATUSUB04', 'DISPUB04', 'Homeless', 'I10_SERVICELINE', 'PAY1', 'PL_NCHS']:
        data.append(pd.get_dummies(filtered_csv[feature], prefix=feature))
    #encoding AGE, I10_NDX, I10_NPR, by normalization
    for feature in ['AGE', 'I10_NDX', 'I10_NPR']:
        #assign the nondigit value to 0
        filtered_csv[feature] = filtered_csv[feature].apply(lambda x: 0 if not str(x).isdigit() else x)
        filtered_csv[feature] = filtered_csv[feature].astype(float)
        data.append((filtered_csv[feature]-filtered_csv[feature].mean())/filtered_csv[feature].std())
    
    #computing the mean and var for nonempty BWT and normalize the nonempty BWT, then assign 0 to empty BWT
    filtered_csv['BWT'] = (filtered_csv['BWT']-filtered_csv['BWT'][~filtered_csv['BWT'].isna()].mean())/filtered_csv['BWT'][~filtered_csv['BWT'].isna()].std()
    filtered_csv['BWT'][filtered_csv['BWT'].isna()] = 0
    
    
    #encoding if PAY2, PAY3 is nonempty by 1, else 0
    for feature in ['PAY2', 'PAY3']:
        data.append(filtered_csv[feature].notnull())
    
    data = pd.concat(data, axis=1)

    train_data = data[filtered_csv['YEAR'] != 2019]
    test_data = data[filtered_csv['YEAR'] == 2019]
    train_label = label[filtered_csv['YEAR'] != 2019]
    test_label = label[filtered_csv['YEAR'] == 2019]
    return train_data, test_data, train_label, test_label


def combined_data_process_i10(args):
    i10_regenerate_process = False
    if i10_regenerate_process:
        csv_path = args.all_data_dir
        csv_file = pd.read_csv(csv_path, header=0)
        filtered_csv =csv_file.dropna(subset=['LOS'])
        # filtered_csv = filtered_csv.head(1000)
        print('in total  ',len(filtered_csv),' patients')
        all_columns = [col for col in csv_file.columns]
        #extract the label
        label = filtered_csv['LOS']
        filtered_csv['combine_icd_list'] = filtered_csv.apply(lambda row: list(row[filtered_csv.columns[filtered_csv.columns.str.startswith('I10_DX')]].values), axis=1)
        # def extract_ga(lst):
        #     for item in lst:
        #         if isinstance(item, str) and item.startswith('Z3A'):
        #             return item[3:]
        #     return None
        # filtered_csv['GA'] = filtered_csv['combine_icd_list'].apply(extract_ga)
        # #adding comorbidity: Elixhauser
        # def get_dx_list(code_list):
        #     out = ee.get_elixhauser(code_list)
        #     output = out['cmrbdt_lst']
        #     return output
        # filtered_csv['cmb_lst'] = filtered_csv['combine_icd_list'].apply(lambda lst: [get_dx_list(str(element)) for element in lst])
        # def clean_list(lst):
        #     return [item[0] for item in lst if len(item)>0]
        #     filtered_csv['cleaned_cmb_lst'] = filtered_csv['cmb_lst'].apply(lambda lst: clean_list(lst))
        # def clean_list_len(lst):
        #     return len([item[0] for item in lst if len(item)>0])
        # filtered_csv['comorb_score'] = filtered_csv['cmb_lst'].apply(lambda lst: clean_list_len(lst))
        # #reducing ICD columns to meaningful CCRS codes, this reduces all icd columns to only ~20 columns
        # ccsr = pd.read_csv('/home/local/zding20/exp2_v2/temp/LoS_Benchmark/data/DXCCSR_v2023-1.csv')
        # maps = dict(zip(ccsr.iloc[:,0].str.strip("'"), ccsr.iloc[:,6].str.strip("'")))
        # def map_to_dict(lst):
        #     x = [maps.get(str(item), 'Unknown') for item in lst]
        #     return [item for item in x if item != 'Unknown']
        # filtered_csv['MappedColumn'] = filtered_csv['combine_icd_list'].apply(map_to_dict)
        # maps12 = dict(zip(ccsr.iloc[:,6].str.strip("'"), ccsr.iloc[:,8].str.strip("'")))
        # for k,v in maps12.items():
        #     if len(v)<6:
        #         maps12[k]=k
        # def map12_to_dict(lst):
        #     x = [maps12.get(str(item), 'Unknown') for item in lst]
        #     return [item for item in x if item != 'Unknown']
        # filtered_csv['MappedColumn12'] = filtered_csv['MappedColumn'].apply(map12_to_dict)
        # cat2 = set(filtered_csv.MappedColumn12.explode().values)
        # for name in cat2:
        #     if str(name)!='Unknown':
        #         filtered_csv[name]=False
        # for i, row in filtered_csv.iterrows():
        #     if str(row.MappedColumn12)!='Unknown':
        #         for name in row.MappedColumn12:
        #             filtered_csv.loc[i,name]=True
                    
        ccsr = pd.read_csv('/home/local/zding20/exp2_v2/temp/LoS_Benchmark/data/DXCCSR_v2023-1.csv')
        filtered_csv['combine_icd_list'] = filtered_csv.apply(lambda row: list(row[filtered_csv.columns[filtered_csv.columns.str.startswith('I10_DX')]].values), axis=1)
        maps = dict(zip(ccsr.iloc[:,0].str.strip("'"), ccsr.iloc[:,6].str.strip("'")))
        maps12 = dict(zip(ccsr.iloc[:,6].str.strip("'"), ccsr.iloc[:,8].str.strip("'")))
        for k,v in maps12.items():
            if len(v)<6:
                maps12[k]=k
        maps23 = dict(zip(ccsr.iloc[:,8].str.strip("'"), ccsr.iloc[:,10].str.strip("'")))
        for k,v in maps23.items():
            if len(v)<6:
                maps23[k]=k
        maps34 = dict(zip(ccsr.iloc[:,10].str.strip("'"), ccsr.iloc[:,12].str.strip("'")))
        for k,v in maps34.items():
            if len(v)<6:
                maps34[k]=k
        def map_to_dict(lst):
            x = [maps.get(str(item), 'Unknown') for item in lst]
            return [item for item in x if item != 'Unknown']
        def map12_to_dict(lst):
            x = [maps12.get(str(item), 'Unknown') for item in lst]
            return [item for item in x if item != 'Unknown']
        def map23_to_dict(lst):
            x = [maps23.get(str(item), 'Unknown') for item in lst]
            return [item for item in x if item != 'Unknown']
        def map34_to_dict(lst):
            x = [maps34.get(str(item), 'Unknown') for item in lst]
            return [item for item in x if item != 'Unknown']
        filtered_csv['MappedColumn'] = filtered_csv['combine_icd_list'].apply(map_to_dict)
        filtered_csv['MappedColumn12'] = filtered_csv['MappedColumn'].apply(map12_to_dict)
        filtered_csv['MappedColumn23'] = filtered_csv['MappedColumn12'].apply(map23_to_dict)
        filtered_csv['MappedColumn34'] = filtered_csv['MappedColumn23'].apply(map34_to_dict)
        #get unique categories from MappedColumn34 without using set func
        cat4 = set(filtered_csv.MappedColumn34.explode().values)
        cat4 = {x for x in cat4 if x==x}
        
        # for name in cat4:
        #     if str(name)!='Unknown':
        #         filtered_csv[name]=False
        #         print(name)
        # print(filtered_csv.shape)
        filtered_csv[list(cat4)] = 0
        for i, row in filtered_csv.iterrows():
            if str(row.MappedColumn34)!='Unknown':
                for name in row.MappedColumn34:
                    filtered_csv.loc[i,name]=1
        
        filtered_csv = filtered_csv[list(cat4)]
        filtered_csv.to_csv('/home/local/zding20/exp2_v2/temp/LoS_Benchmark/data/combined_2016to2019_i10.csv')
    else:
        filtered_csv = pd.read_csv('/home/local/zding20/exp2_v2/temp/LoS_Benchmark/data/combined_2016to2019_i10.csv')
    columns = filtered_csv.columns.tolist()
    label = filtered_csv['LOS']
    data = filtered_csv
        
    if args.normalization:
        label = label / 10
        
    train_data = data[filtered_csv['YEAR'] != 2019]
    test_data = data[filtered_csv['YEAR'] == 2019]
    train_label = label[filtered_csv['YEAR'] != 2019]
    test_label = label[filtered_csv['YEAR'] == 2019]
    return train_data, test_data, train_label, test_label


def combined_data_process_all(args):
    data_i10prc = pd.read_csv('../combined_2016to2019_i10prc.csv')
    data_i10prc = data_i10prc.drop(['LOS', 'YEAR'], axis=1)
    data_demographic = pd.read_csv(
        '../combined_2016to2019_demographic.csv')
    # concatenate the tables
    data = pd.concat([data_i10prc, data_demographic], axis=1)

    filtered_csv = data

    # remove los and year for data
    data = data.drop(columns=['LOS', 'YEAR'])
    # dropnan cols
    data = data.dropna(axis=1)
    # drop Unnamed: 0
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    label = filtered_csv['LOS']

    common_features = ['ZIP', 'RACE', 'AGE', 'FEMALE', 'ATYPE', 'READMIT', 'I10_NDX', 'I10_NPR', 'MARITALSTATUSUB04',
                       'BWT', 'DISPUB04', 'Homeless',
                       'I10_SERVICELINE', 'PAY1', 'PAY2', 'PAY3', 'PL_NCHS', 'YEAR']
    filtered_csv = filtered_csv[common_features]
    data = []

    for feature in ['RACE', 'FEMALE', 'ATYPE', 'READMIT', 'MARITALSTATUSUB04', 'DISPUB04', 'Homeless',
                    'I10_SERVICELINE', 'PAY1', 'PL_NCHS']:
        data.append(pd.get_dummies(filtered_csv[feature], prefix=feature))
    # encoding AGE, I10_NDX, I10_NPR, by normalization
    for feature in ['AGE', 'I10_NDX', 'I10_NPR']:
        # assign the nondigit value to 0
        filtered_csv[feature] = filtered_csv[feature].apply(lambda x: 0 if not str(x).isdigit() else x)
        filtered_csv[feature] = filtered_csv[feature].astype(float)
        data.append((filtered_csv[feature] - filtered_csv[feature].mean()) / filtered_csv[feature].std())

    # computing the mean and var for nonempty BWT and normalize the nonempty BWT, then assign 0 to empty BWT
    filtered_csv['BWT'] = (filtered_csv['BWT'] - filtered_csv['BWT'][~filtered_csv['BWT'].isna()].mean()) / \
                          filtered_csv['BWT'][~filtered_csv['BWT'].isna()].std()
    filtered_csv['BWT'][filtered_csv['BWT'].isna()] = 0

    # encoding if PAY2, PAY3 is nonempty by 1, else 0
    for feature in ['PAY2', 'PAY3']:
        data.append(filtered_csv[feature].notnull())

    data = pd.concat(data, axis=1)

    train_data = data[filtered_csv['YEAR'] != 2019]
    test_data = data[filtered_csv['YEAR'] == 2019]
    train_label = label[filtered_csv['YEAR'] != 2019]
    test_label = label[filtered_csv['YEAR'] == 2019]

    return train_data, test_data, train_label, test_label
