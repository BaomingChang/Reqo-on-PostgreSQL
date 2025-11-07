import os
import numpy as np
from numpy import nanmean
from torch import optim
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from Models.reqo_model import Reqo
from Utils.loss import *
from Utils.evaluate import get_qerror_and_spearman, get_plansubop_and_runtime, write_results_to_file, plot_runtimes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(dataset, batch_size, shuffle_or_not):
    labels = [row[2] for row in dataset]
    data = [Data(x=torch.FloatTensor(row[0]),
                 edge_index=sort_edge_index(torch.LongTensor(row[1]).t()),
                 y=row[2]) for row in dataset]
    dataset_loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=shuffle_or_not)
    return dataset_loader, max(labels), min(labels)

def train(dbname, reqo_config, k_i, trainset, testset, save_path, query_plans_index_num_i, query_postgres_cost_i, save_model):
    batch_size = reqo_config["batch_size"]
    table_columns_number = np.load(f'Data/{dbname}/database_statistics/table_columns_number.npy')

    train_loader, max_sup_train_label, min_sup_train_label = load_dataset(trainset, batch_size,  True)
    test_loader, max_sup_test_label, min_sup_test_label = load_dataset(testset, batch_size, False)
    testset_size = len(testset)
    max_label_log = math.log(max(max_sup_train_label, max_sup_test_label)+1)
    min_label_log = math.log(min(min_sup_train_label, min_sup_test_label)+1)

    # load model
    encoder_params = {k: v for k, v in reqo_config.items() if k.startswith("encoder_")}
    estimator_params = {k: v for k, v in reqo_config.items() if k.startswith("estimator_")}
    encoder_params["encoder_table_num"] = len(table_columns_number)
    encoder_params["encoder_column_num"] = sum(table_columns_number)

    model = Reqo(encoder_params=encoder_params, estimator_params=estimator_params)
    model = model.to(device)

    # criteon_logmse = LogMSELoss()
    criteon_data_uncertainty = DataUncertaintyLoss()
    criteon_ranking = PairRankingLoss()
    optimizer = optim.Adam(model.parameters(), lr=reqo_config["learning_rate"])

    epochs = 100
    early_stop = 0
    best_test_perf =float('inf')
    for epoch in range(epochs):
        if early_stop >= 20:
            break
        # Train
        model.train()
        step = 0
        train_loss_all = 0
        with tqdm(train_loader, desc=f'Fold {k_i} Epoch: {epoch + 1} Training ', unit='batch') as train_loader:
            for batch in train_loader:
                batch.to(device)
                optimizer.zero_grad()
                batch_train_pred, batch_train_va, batch_train_iv = model(batch, table_columns_number)
                train_data_uncertainty_loss = criteon_data_uncertainty(batch_train_pred, batch_train_va, batch.y.float(), max_label_log, min_label_log)
                train_ranking_loss = criteon_ranking(batch_train_iv, batch.y.float(), max_label_log, min_label_log)
                train_loss = train_data_uncertainty_loss + train_ranking_loss
                train_loss.backward()
                optimizer.step()

                batch_graph_num = batch.num_graphs
                train_loss_all += train_loss.item()*batch_graph_num
                step += batch_graph_num
            avg_train_loss = float(train_loss_all/step)

        # Evaluate
        model.eval()
        step = 0
        test_loss_all = 0
        actual_latency = torch.zeros(testset_size, device=device)
        pred_ev = torch.zeros(testset_size, device=device)
        pred_va = torch.zeros(testset_size, device=device)
        pred_iv = torch.zeros(testset_size, device=device)
        with tqdm(test_loader, desc=f'Fold {k_i} Epoch: {epoch + 1} Testing ', unit='batch') as test_loader:
            for batch in test_loader:
                batch = batch.to(device)
                with torch.no_grad():
                    batch_test_pred, batch_test_va, batch_test_iv = model(batch, table_columns_number)
                test_data_uncertainty = criteon_data_uncertainty(batch_test_pred, batch_test_va, batch.y.float(), max_label_log, min_label_log)
                test_ranking_loss = criteon_ranking(batch_test_iv, batch.y.float(), max_label_log, min_label_log)
                test_loss = test_data_uncertainty + test_ranking_loss
                batch_graph_num = batch.num_graphs
                test_loss_all += test_loss.item()*batch_graph_num
                end = step + batch_graph_num
                pred_ev[step:end] = batch_test_pred.view(-1)
                pred_va[step:end] = batch_test_va.view(-1)
                pred_iv[step:end] = batch_test_iv.view(-1)
                actual_latency[step:end] = batch.y.view(-1)
                step = end
            avg_test_loss = float(test_loss_all/step)

        pred_ev = pred_ev.cpu().numpy()
        pred_iv = pred_iv.cpu().numpy()
        actual_latency = actual_latency.cpu().numpy()

        cost_estimation_results = get_qerror_and_spearman(pred_ev, actual_latency, max_label_log, min_label_log)
        robustness_results, runtime_per_query = get_plansubop_and_runtime(pred_iv, actual_latency,
                                                                          query_postgres_cost_i,
                                                                          query_plans_index_num_i)
        print(f'Fold {k_i} Epoch {epoch + 1}: train_loss: {avg_train_loss}, test_loss: {avg_test_loss}, spearmancorrelation: {cost_estimation_results[-1]}, optimal_runtime_ratio: {robustness_results[11]}')

        # # Early stop based on test loss
        # if avg_test_loss < best_test_perf:
        #     best_test_perf = avg_test_loss
        #     best_model = model.state_dict()
        #     best_cost_estimation_results = cost_estimation_results
        #     best_robustness_results = robustness_results
        #     best_runtime_per_query = runtime_per_query
        #     early_stop = 0
        # else:
        #     early_stop += 1

        # Early stop based on optimal runtime ratio
        if robustness_results[11] < best_test_perf:
            best_test_perf = robustness_results[11]
            best_model = model.state_dict()
            best_cost_estimation_results = cost_estimation_results
            best_robustness_results = robustness_results
            best_runtime_per_query = runtime_per_query
            early_stop = 0
        else:
            early_stop += 1

    cost_estimation_results = best_cost_estimation_results
    robustness_results = best_robustness_results
    runtime_per_query = best_runtime_per_query

    print(f'Fold {k_i}: test results: qerror_median: {cost_estimation_results[4]} qerror_top99mean: {cost_estimation_results[2]}, spearman_correlation: {cost_estimation_results[-1]}, subop_median: {robustness_results[4]}, subop_top99mean: {robustness_results[2]}, model_to_postgresql_runtime_ratio: {robustness_results[10]}, model_to_optimal_runtime_ratio: {robustness_results[11]}')
    os.makedirs(save_path, exist_ok=True)
    write_results_to_file(cost_estimation_results + robustness_results, expl_or_not=False, filename=save_path + 'reqo_fold_' + str(k_i) + '_results.txt')
    if save_model:
        torch.save(best_model, save_path + 'reqo_fold_' + str(k_i) + '_model.pth')
    return cost_estimation_results+robustness_results, runtime_per_query

def k_fold_train(dbname, reqo_config, k=10, save_model=False):
    save_path = f'Results/{dbname}/'
    os.makedirs(save_path, exist_ok=True)
    dataset = np.load(f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_plans_dataset.npy', allow_pickle=True)
    query_plans_index_num = np.load(f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_plans_index_num.npy', allow_pickle=True)
    query_postgres_cost = np.load(f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_plans_postgres_cost.npy', allow_pickle=True)
    k_sample_num = round(len(query_plans_index_num) / k)

    all_results = []
    all_postgres_runtimes, all_reqo_runtimes, all_optimal_runtimes = [], [], []
    for k_i in range(0, k):
        sample_q_num1 = k_sample_num*k_i
        sample_q_num2 = k_sample_num*(k_i+1)
        sample_p_num1 = sum(query_plans_index_num[:sample_q_num1])
        sample_p_num2 = sample_p_num1 + sum(query_plans_index_num[sample_q_num1:sample_q_num2])

        trainset = np.concatenate((dataset[:sample_p_num1], dataset[sample_p_num2:]), axis=0)
        testset = dataset[sample_p_num1:sample_p_num2]

        query_plans_index_num_i = query_plans_index_num[sample_q_num1:sample_q_num2]
        query_postgres_cost_i = query_postgres_cost[sample_q_num1:sample_q_num2]

        results, runtime_per_query = train(dbname, reqo_config, k_i+1, trainset, testset, save_path+'fold_'+str(k_i+1)+'/', query_plans_index_num_i, query_postgres_cost_i, save_model)
        all_results.append(results)
        all_postgres_runtimes.extend(runtime_per_query[0])
        all_reqo_runtimes.extend(runtime_per_query[1])
        all_optimal_runtimes.extend(runtime_per_query[2])

    write_results_to_file(nanmean(np.array(all_results), axis=0), expl_or_not=False, filename=save_path + 'reqo_avg_results.txt')
    plot_runtimes(all_postgres_runtimes, all_reqo_runtimes, all_optimal_runtimes, save_path + 'reqo_runtime_performance.png')


if __name__ == '__main__':
    dbname = 'stats'
    reqo_config = {'batch_size': 256, 'learning_rate': 0.001,
              'encoder_attention_heads': 8, 'encoder_conv_layers': 4, 'encoder_gnn_embedding_dim': 256, 'encoder_gnn_dropout_rate': 0.1, 'encoder_dirgnn_alpha': 0.3, 'encoder_node_type_embedding_dim': 16, 'encoder_column_embedding_dim': 8,
              'explainer_fcn_layers': 4, 'explainer_explanation_embedding_dim': 512, 'explainer_fcn_dropout_rate': 0.1,
              'estimator_fcn_layers': 4, 'estimator_estimation_embedding_dim': 512, 'estimator_fcn_dropout_rate': 0.1}
    k_fold_train(dbname, reqo_config, k=10)
