'''this is the main file for LoS prediction benchmark, version 11/08/2023'''

import argparse
import os
from utils import dataset, models, train_utils
parser = argparse.ArgumentParser(description='LoS prediction benchmark')
parser.add_argument('--all_data_dir', type=str, default='/home/local/zding20/exp2_v2/temp/LoS_Benchmark/data/combined_2016to2019.csv',
                    help='loading directory of the data')
parser.add_argument('--dataset_regenerate', type=bool, default=False,
                    help='whether to regenerate the cohert from the all data')
# parser.add_argument('--cohert_dir', type=str, default='E:\\2023 fall courseworks\\HSI2\\program\\data\\MD_2020_SIDC\\MD_SIDC_2020_CORE_filtered.csv',
#                     help='cohert loading and saving directory')
parser.add_argument('--model', type=str, choices= ['Linear_R','Logistic_R','RF','SVM','XGBoost','MLP'], default='MLP',
                    help='model to use')
parser.add_argument('--output_dir', type=str, default='/home/local/zding20/exp2_v2/temp/LoS_Benchmark/output',
                    help='output directory of the results')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--validation', type=bool, default=False,
                    help='recommand for sklearn based models to be False, since sklearn has its own validation method')
parser.add_argument('--exp_process', type=str, default='randomized_search',choices=['randomized_search','cv','normal'],
                    help='types of experiment process')
parser.add_argument('--normalization', type=bool, default=False,
                    help='normalization of LOS')
parser.add_argument('--dataset_loading_manner', type=str, default='combined',choices=['combined','simple'],
                    help='method to load the dataset')
parser.add_argument('--input_type', type=str, default='demographic',choices=['i10','demographic'],
                    help='method to load the dataset')
def train_process():
    args = parser.parse_args()
    args.all_data_dir = '/home/local/zding20/exp2_v2/temp/LoS_Benchmark/data/combined_2016to2019.csv'
    print(args)
    
    #part 1: load data and dataset split
    if args.dataset_loading_manner == 'simple':
        data, label = dataset.simple_loading_process(args)
        if args.validation:
            train_data, val_data, test_data, train_label, val_label, test_label = dataset.tvt_split(args,data,label)
            data_list = [train_data, val_data, test_data, train_label, val_label, test_label]
        else:
            train_data, test_data, train_label, test_label = dataset.tt_split(args,data,label)
            data_list = [train_data, test_data, train_label, test_label]
            
    if args.dataset_loading_manner == 'combined':
        if args.input_type =='demographic':
            train_data, test_data, train_label, test_label = dataset.combined_data_process(args)
            data_list = [train_data, test_data, train_label, test_label]
        elif args.input_type == 'i10':
            train_data, test_data, train_label, test_label = dataset.combined_data_process_i10(args)
            data_list = [train_data, test_data, train_label, test_label]
    if args.model not in ['MLP']:
        #part 2: model creation
        model = models.model_creation(args)
        #part 3: model training
        train_utils.train_process_conventional(args, model, data_list, exp_process='randomized_search')
    if args.model in ['MLP']:
        #use torch for MLP
        args.input_dim = train_data.shape[1]
        model = models.model_creation(args)
        train_utils.train_process_deeplearning(args, model, data_list, exp_process='randomized_search')


if __name__ == '__main__':
    train_process()
    
