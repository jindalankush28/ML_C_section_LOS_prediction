import sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint
import torch
#TODO: remove the MLP one into torch version, current version is stupid

def model_creation(args):
    if args.model == 'Linear_R':
        model = LinearRegression()
    elif args.model == 'Logistic_R':
        model = LogisticRegression()
    elif args.model == 'RF':
        # model = RandomForestRegressor()
        model = RandomForestClassifier()
    elif args.model == 'SVM':
        # model = SVR()
        model = SVC()
    elif args.model == 'XGBoost':
        # model = GradientBoostingRegressor()
        model = GradientBoostingClassifier()
    elif args.model == 'MLP':
        model = torch.nn.Sequential(
            torch.nn.Linear(args.input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
    return model

def randomized_search(model):

    scoring = {'accuracy': make_scorer(accuracy_score),
               'f1': make_scorer(f1_score),
               'roc_auc': make_scorer(roc_auc_score)}
    
    if type(model).__name__ == 'LinearRegression':
        param_distributions = {
            'positive': [True, False],
            'fit_intercept': [True, False]
        }
        param_search = RandomizedSearchCV(model, param_distributions)

     elif type(model).__name__ == 'LogisticRegression':
        param_distributions = {
            'C': [0.01, 0.05, 1, 0.2, 0.3, 0.5, 0.75, 1, 2, 5, 10],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            # 'max_iter': [50, 75, 100, 125, 150], default = 100
            'max_iter': [2, 3, 4, 5],
            'solver': ['saga']
        }
        param_search = RandomizedSearchCV(model, param_distributions, n_iter=10, scoring=scoring, refit='accuracy')

    elif type(model).__name__ in ['RandomForestRegressor', 'RandomForestClassifier']:
        param_distributions = {
            'n_estimators': [50 ,75, 100, 125, 150],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 3, 5, 10],
            'min_samples_leaf': [1, 2, 3, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        param_search = RandomizedSearchCV(model, param_distributions, n_iter=20, scoring=scoring, refit='accuracy')

    elif type(model).__name__ in ['SVR', 'SVC']:
        def custom_param_sampler_svr(n_iter):
            params_list = []
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            for _ in range(n_iter):
                kernel = np.random.choice(kernels)
                params = {
                    'C': [loguniform(1, 100).rvs()],
                    'gamma': [loguniform(0.001, 0.1).rvs()],
                    'kernel': [kernel],
                    'max_iter': [5000],
                    'verbose': [False],
                }
                if kernel[0] == 'poly':
                    params['degree'] = [randint(2, 5).rvs()]
                if kernel[0] in ['poly', 'sigmoid']:
                    params['coef0'] = [loguniform(0.01, 10).rvs()]
                params_list.append(params)
            return params_list
        n_iter = 20
        params_list = custom_param_sampler_svr(n_iter)  
        scoring['mean_squared_error'] = make_scorer(mean_squared_error, greater_is_better=False)
        param_search = RandomizedSearchCV(model, params_list, n_iter=n_iter,error_score='raise', scoring=scoring, refit='mean_squared_error')
        
    elif type(model).__name__ in ['GradientBoostingRegressor', 'GradientBoostingClassifier']:
        param_distributions = {
            'n_estimators': [50 ,75, 100, 125, 150],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 3, 5, 10],
            'min_samples_leaf': [1, 2, 3, 4],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.5, 0.75, 0.1]
        }
        param_search = RandomizedSearchCV(model, param_distributions, n_iter=20,scoring=scoring, refit='accuracy')

    elif type(model).__name__ == 'MLPRegressor':
        param_distributions = {
            'hidden_layer_sizes': [(64,), (128, 64), (128, 128), (64, 64, 64),(128, 128, 128)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'alpha': [0.001, 0.01, 0.1, 1],
            'solver': ['sgd', 'adam'],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'max_iter': [200, 500, 1000],
            'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
            'early_stopping': [True],
            'verbose': [False]
        }

        param_search = RandomizedSearchCV(model, param_distributions, n_iter=2,error_score='raise', scoring='neg_mean_squared_error')
    return param_search

def custom_random_search_mlp(n_iter=20):
    params_list = []
    param_distributions = {
            'num_epochs': [100],
            # 'hidden_layers': [(64,32), (128, 64), (128, 128), (64, 64, 64),(128, 128, 128)],
            'hidden_layers': [(512, 1024, 128)],
            'activation': ['relu', 'tanh', 'sigmoid'],
            'dropout': [0, 0.1, 0.2, 0.3],
        }
    for _ in range(n_iter):
        num_epochs = np.random.choice(param_distributions['num_epochs'])
        hidden_layers = param_distributions['hidden_layers'][np.random.choice(np.arange(len(param_distributions['hidden_layers'])))]
        activation = np.random.choice(param_distributions['activation'])
        dropout = np.random.choice(param_distributions['dropout'])
        params = {
            'num_epochs': [num_epochs],
            'hidden_layers': [hidden_layers],
            'activation': [activation],
            'dropout': [dropout],
        }
        params_list.append(params)
    return params_list

def construct_mlp(params,input_dim):
    activation_layer = {
        'relu': torch.nn.ReLU(),
        'tanh': torch.nn.Tanh(),
        'sigmoid': torch.nn.Sigmoid(),
    }
    
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, params['hidden_layers'][0][0]),
        # torch.nn.BatchNorm1d(params['hidden_layers'][0][0]),
        activation_layer[params['activation'][0]],
        torch.nn.Dropout(params['dropout'][0]),
        torch.nn.Linear(params['hidden_layers'][0][0], params['hidden_layers'][0][1]),
        # torch.nn.BatchNorm1d(params['hidden_layers'][0][1]),
        activation_layer[params['activation'][0]],
    )
    if len(params['hidden_layers'][0]) == 3:
        model.add_module('hidden_layer_3', torch.nn.Linear(params['hidden_layers'][0][1], params['hidden_layers'][0][2]))
        # model.add_module('batch_norm_3', torch.nn.BatchNorm1d(params['hidden_layers'][0][2]))
        model.add_module('activation_3', activation_layer[params['activation'][0]])
    model.add_module('output_layer', torch.nn.Linear(params['hidden_layers'][0][-1], 2))
    return model

