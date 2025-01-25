import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, permutation_test_score, learning_curve, LearningCurveDisplay
from utils.models import randomized_search,custom_random_search_mlp,construct_mlp
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
import joblib
from torch.utils.data import TensorDataset, DataLoader
import torch

torch.set_num_threads(10)
def train_process_conventional(args, model, data_list,exp_process='randomized_search'):
    assert len(data_list) == 4, 'data_list should be [train_data, test_data, train_label, test_label]'
    train_data, test_data, train_label, test_label = data_list
    train_label = one_hot_encoding(train_label)
    test_label = one_hot_encoding(test_label)
    exp_dir = os.path.join(args.output_dir,args.model,exp_process)
    os.makedirs(exp_dir, exist_ok=True)
    if exp_process == 'cv':
        scores = cross_val_score(model, train_data, train_label, cv=5)
        test_pred = model.predict(test_data)
        test_loss = sklearn.metrics.mean_squared_error(test_label, test_pred)
        print('train loss with CV: ', scores.mean())
        print('test loss: ', test_loss)
    if exp_process == 'randomized_search':
        rs = randomized_search(model)
        rs.fit(train_data, train_label)
        model = rs.best_estimator_

        '''
        # Linear_R Mean and SD are manually calculated since sklearn do not support post processing of prediction
        kf = KFold(n_splits=5)
        accuracy_scores = []
        f1_scores = []
        roc_auc_scores = []

        for train_index, test_index in kf.split(train_data):
            train_d, test_d = train_data.iloc[train_index], train_data.iloc[test_index]
            train_l, test_l = train_label.iloc[train_index], train_label.iloc[test_index]

            # Initialize and fit the model
            model.fit(train_d, train_l)

            # Make predictions and apply threshold
            test_pred = model.predict(test_d)
            test_pred = np.where(test_pred > 0.5, 1, 0)

            # Calculate scores
            accuracy_scores.append(accuracy_score(test_l, test_pred))
            f1_scores.append(f1_score(test_l, test_pred))
            roc_auc_scores.append(roc_auc_score(test_l, test_pred))

        # Calculate mean and standard deviation for each metric
        accuracy_mean, accuracy_std = np.mean(accuracy_scores), np.std(accuracy_scores)
        f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
        roc_auc_mean, roc_auc_std = np.mean(roc_auc_scores), np.std(roc_auc_scores)

        print(f"Accuracy: Mean = {accuracy_mean}, Std = {accuracy_std}")
        print(f"F1 Score: Mean = {f1_mean}, Std = {f1_std}")
        print(f"ROC AUC Score: Mean = {roc_auc_mean}, Std = {roc_auc_std}")
        '''
        
        cv_results = pd.DataFrame(rs.cv_results_)
        cv_results.to_csv(os.path.join(exp_dir, 'cv_results.csv'))
        print('Best Model: ', model)
        model.fit(train_data, train_label)
        train_pred = model.predict(train_data)
        test_pred = model.predict(test_data)
        train_loss = sklearn.metrics.mean_squared_error(train_label, train_pred)
        test_loss = sklearn.metrics.mean_squared_error(test_label, test_pred)
        print('train loss: ', train_loss)
        print('test loss: ', test_loss)
        # save the model
        model_path = os.path.join(exp_dir, 'model.pkl')
        joblib.dump(model, model_path)

        # scoring for best model and SD from randomized search
        index = rs.best_index_
        acc = cv_results['mean_test_accuracy'][index]
        std_acc = cv_results['std_test_accuracy'][index]
        print(f'Test_ACC: {acc}({std_acc})')
        f1 = cv_results['mean_test_f1'][index]
        std_f1 = cv_results['std_test_f1'][index]
        print(f'Test_F1: {f1}({std_f1})')
        # test_confusion_matrix = sklearn.metrics.confusion_matrix(test_label, test_pred)
        auc = cv_results['mean_test_roc_auc'][index]
        std_auc = cv_results['std_test_roc_auc'][index]
        print(f'Test_AUC: {auc}({std_auc})')
        
        # plot learning curve of loss
        train_sizes, train_scores, test_scores = learning_curve(model, train_data, train_label)
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        plt.title('Learning curve')
        plt.xlabel('Training Data Size')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(os.path.join(exp_dir, 'learning_curve.png'))
        plt.show()
        plt.clf()
        
        # # permutation test score of best model
        # data_combined = np.concatenate([train_data, test_data])
        # label_combined = np.concatenate([train_label, test_label])
        # score, permutation_scores, pvalue = permutation_test_score(model, data_combined, label_combined,
        #                                                            scoring="neg_mean_squared_error",
        #                                                            n_permutations=20)
        # fig, ax = plt.subplots()
        # print(f"Model Permutation Test Score: {score}")
        # print(f"P-value: {pvalue}")
        # ax.axvline(score, ls="--", color="r")
        # ax.hist(permutation_scores, 20, label='Permutation scores',
        #          edgecolor='black')
        # score_label = f"Score on original\ndata: {score:.2f}\n(p-value: {pvalue:.3f})"
        # ax.text(0.14, 7.5, score_label, fontsize=12)
        # ax.set_xlabel("Accuracy score")
        # ax.set_ylabel("Probability density")
        # plt.show()
        # plt.savefig(os.path.join(exp_dir, 'permutation_test_score.png'))
        # plt.clf()
def train_process_deeplearning(args, model, data_list,exp_process='randomized_search'):
    assert len(data_list) == 4, 'data_list should be [train_data, test_data, train_label, test_label]'
    train_data, test_data, train_label, test_label = data_list
    exp_dir = os.path.join(args.output_dir,args.model,exp_process)
    os.makedirs(exp_dir, exist_ok=True)
    if exp_process == 'cv':
        #TODO- modifying this part for torch mode
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        for i, (train_index, val_index) in enumerate(kfold.split(train_data)):
            train_data, val_data = train_data[train_index], train_data[val_index]
            train_label, val_label = train_label[train_index], train_label[val_index]
            #generate dataloader
            train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
            val_dataset = TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_label))
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
            #train the model
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = torch.nn.MSELoss()
            train_loss_list = []
            val_loss_list = []
            for epoch in range(20):
                train_loss = 0
                for data, label in train_loader:
                    train_loss += train_step(model, data, label, optimizer, loss_fn)
                train_loss_list.append(train_loss/len(train_loader))
                val_loss = 0
                for data, label in val_loader:
                    val_loss += valid_step(model, data, label, loss_fn)
                val_loss_list.append(val_loss/len(val_loader))
                
            #test loss
            test_pred = model(torch.from_numpy(test_data))
            test_loss = loss_fn(test_pred, torch.from_numpy(test_label))
            print('train loss with CV: ', train_loss_list[-1])
            print('val loss with CV: ', val_loss_list[-1])
            print('test loss: ', test_loss)
            
    if exp_process == 'randomized_search':
        # generate the params list for randomized search
        params_list = custom_random_search_mlp(n_iter=1)
        train_val_result = []
        test_result = []
        train_data_all, test_data_all, train_label_all, test_label_all = data_list
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        for params in params_list:
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            train_val_result_current_params = []
            test_result_current_params = []
            for i, (train_index, val_index) in enumerate(kfold.split(train_data_all)):
                train_data, val_data = np.array(train_data_all.values[train_index],dtype = np.float16), np.array(train_data_all.values[val_index],dtype = np.float16)
                train_label, val_label = np.array(train_label_all.values[train_index],dtype=np.float16), np.array(train_label_all.values[val_index],dtype = np.float16)
                #compute the number for each class and generate the weight
                # channel_weight = np.zeros((6,))
                # for i in range(6):
                #     channel_weight[i] = train_label.shape[0]/np.sum(train_label==i)
                channel_weight = None
                #generate dataloader
                train_label_onehot = one_hot_encoding(train_label)
                val_label_onehot = one_hot_encoding(val_label)
                # train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
                # val_dataset = TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_label))
                train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label_onehot))
                val_dataset = TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_label_onehot))
                train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)
                #construct the model
                model = construct_mlp(params, args.input_dim).to(device)
                #train the model
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
                loss_fn = torch.nn.MSELoss()
                # loss_fn = torch.nn.MSELoss(reduction='none')
                train_loss_list = []
                val_loss_list = []
                for epoch in range(20):
                    train_loss = 0
                    for data, label in train_loader:
                        train_loss += train_step(model, data, label, optimizer, loss_fn, device, channel_weight=channel_weight)
                    train_loss_list.append(train_loss/len(train_loader))
                    val_loss = 0
                    for data, label in val_loader:
                        val_loss += valid_step(model, data, label, loss_fn, device, channel_weight=channel_weight)
                    val_loss_list.append(val_loss/len(val_loader))
                train_val_result_current_params.append([train_loss_list, val_loss_list])
                
                test_pred = model(torch.from_numpy(np.array(test_data_all.values,dtype = np.float32)).to(device))
                # test_acc = sklearn.metrics.accuracy_score(torch.argmax(test_pred, dim=1).cpu().numpy(), np.array(test_label_all.values,dtype = np.float32))
                test_acc = sklearn.metrics.accuracy_score(torch.argmax(test_pred, dim=1).cpu().numpy(), np.array(test_label_all.values>3,dtype = np.float32))
                test_f1 = sklearn.metrics.f1_score(torch.argmax(test_pred, dim=1).cpu().numpy(), np.array(test_label_all.values>3,dtype = np.float32))
                test_confusion_matrix = sklearn.metrics.confusion_matrix(torch.argmax(test_pred, dim=1).cpu().numpy(), np.array(test_label_all.values>3,dtype = np.float32))
                test_auc = sklearn.metrics.roc_auc_score(torch.argmax(test_pred, dim=1).cpu().numpy(), np.array(test_label_all.values>3,dtype = np.float32))
                
                # test_loss = loss_fn(test_pred, torch.from_numpy(np.array(test_label.values,dtype = np.float32)).to(device))
                test_result_current_params.append(test_loss.item())
            train_val_result.append(train_val_result_current_params)
            test_result.append(test_result_current_params)
            np.save(os.path.join(exp_dir, 'train_val_result.npy'), np.array(train_val_result))
            np.save(os.path.join(exp_dir, 'test_result.npy'), np.array(test_result))

def one_hot_encoding(label):
    # label_onehot = np.zeros((label.shape[0],6))
    # label_onehot[label<2,0] = 1
    # label_onehot[label==2,1] = 1
    # label_onehot[label==3,2] = 1
    # label_onehot[label==4,3] = 1
    # label_onehot[label==5,4] = 1
    # label_onehot[label>5,5] = 1
    
    label_onehot = np.zeros((label.shape[0],2))
    label_onehot[label<=3,0] = 1
    label_onehot[label>3,1] = 1
    return label_onehot
        
def train_step(model, data, label, optimizer, loss_fn,device,channel_weight=None):
    model.train()
    optimizer.zero_grad()
    pred = model(data.float().to(device))
    if channel_weight is not None:
        channel_weight = torch.clip(torch.from_numpy(channel_weight).float().to(device),0,20)
        loss = torch.multiply(loss_fn(pred, label.float().to(device)),channel_weight).sum(axis = 1).mean()
    else:
        loss = loss_fn(pred, label.float().to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

def valid_step(model, data, label, loss_fn,device,channel_weight=None):
    model.eval()
    pred = model(data.float().to(device))
    if channel_weight is not None:
        channel_weight = torch.clip(torch.from_numpy(channel_weight).float().to(device),0,20)
        loss = torch.multiply(loss_fn(pred, label.float().to(device)),channel_weight).sum(axis = 1).mean()
    else:
        loss = loss_fn(pred, label.float().to(device))
    return loss.item()
    
