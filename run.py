import sys
import random
import os
import numpy as np
import re
from collections import defaultdict
# sys.path.append('/home/.../SmileGNN/')  # add the env path
from sklearn.model_selection import train_test_split, StratifiedKFold
from main import train
from datetime import datetime
# from SmileGNN.layers.feature import *
import logging
import config
import pandas as pd
from utils import pickle_dump, format_filename, write_log, pickle_load

logging.basicConfig(level=logging.INFO, format= '[%(asctime)s] [%(pathname)s:%(lineno)d] [%(levelname)s] - %(message)s',
     datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('log/run.log')]) 
logger = logging.getLogger(__name__)

def read_entity2id_file(file_path: str, drug_vocab: dict, entity_vocab: dict,dataset:str):
    """
    Reads a file containing entity-to-ID mappings and returns a dictionary
    mapping knowledge graph paths to new identifiers.
    
    Args:
        file_path (str): Path to the entity2id file.
        drug_vocab (dict): Dictionary mapping drug names to new identifiers.
        entity_vocab (dict): Dictionary mapping entity names to new identifiers.
        dataset (str): Dataset name.
    Returns:
        entity2id (dict): Dictionary mapping entity names to new identifiers.

    """
    print(f'Logging Info - Reading entity2id file: {file_path}')
    assert len(drug_vocab) == 0 and len(entity_vocab) == 0
    entity2id = {}
    with open(file_path, encoding='utf8') as reader:
        for idx, line in enumerate(reader):
            if (idx == 0): continue
            drug, entity = line.strip().split('\t') if dataset == 'kegg' else line.strip().split(' ') # only kegg and pdd
            drug_vocab[entity] = len(drug_vocab)
            entity_vocab[entity] = len(entity_vocab)
            entity2id[drug] = drug_vocab[entity]
    return entity2id

def read_approved_example_file(file_path: str, separator: str, drug_vocab: dict):
    '''
    Reads a file containing approved drug-drug interactions and returns a list of examples.
    Args:
        file_path (str): Path to the approved drug-drug interaction file.
        separator (str): Separator used in the file.
        drug_vocab (dict): Dictionary mapping drug names to new identifiers.
    Returns:
        examples (np.array): Array of examples of drug interaction matrix [drug1_id, drug2_id, 1/0]
    '''
    print(f'Logging Info - Reading example file: {file_path}')
    assert len(drug_vocab) > 0
    examples = []
    with open(file_path, encoding='utf8') as reader:
        for idx, line in enumerate(reader):
            d1, d2, flag = line.strip().split(separator)[:3]
            if d1 not in drug_vocab or d2 not in drug_vocab:
                continue
            if d1 in drug_vocab and d2 in drug_vocab:
                examples.append([drug_vocab[d1], drug_vocab[d2], int(flag)])

    examples_matrix = np.array(examples)
    logger.info(f'size of example: {examples_matrix.shape}')
    return examples_matrix

def read_kg(file_path: str, entity_vocab: dict, relation_vocab: dict, neighbor_sample_size: int):
    '''
    Reads a file containing knowledge graph triples and returns a dictionary
    mapping entity IDs to a list of (neighbor, relation) tuples.
    Args:
        file_path (str): Path to the knowledge graph file.
        entity_vocab (dict): Dictionary mapping entity names to new identifiers.
        relation_vocab (dict): Dictionary mapping relation names to new identifiers.
        neighbor_sample_size (int): Number of neighbors to sample for each entity.
    Returns:
        kg (dict): Dictionary mapping entity IDs to a list of (neighbor, relation) tuples.
    '''
    print(f'Logging Info - Reading kg file: {file_path}')

    kg = defaultdict(list)
    with open(file_path, encoding='utf8') as reader:
        count = 0
        for line in reader:
            if count == 0:
                count += 1
                continue
            head, tail, relation = line.strip().split(' ')

            if head not in entity_vocab:
                entity_vocab[head] = len(entity_vocab)
            if tail not in entity_vocab:
                entity_vocab[tail] = len(entity_vocab)
            if relation not in relation_vocab:
                relation_vocab[relation] = len(relation_vocab)

            # undirected graph
            kg[entity_vocab[head]].append((entity_vocab[tail], relation_vocab[relation]))
            kg[entity_vocab[tail]].append((entity_vocab[head], relation_vocab[relation]))
    
    logger.info(f'Num of entities: {len(entity_vocab)}, '
          f'Num of relations: {len(relation_vocab)}')

    logger.info(f'Constructing adjacency matrix with {neighbor_sample_size}')
    n_entity = len(entity_vocab)
    adj_entity = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)
    adj_relation = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)

    np_neighbor_ctr = 0
    # If the number of neighboring nodes is greater than neighbor_sample_size, randomly select neighbor_sample_size of them
    for entity_id in range(n_entity):
        all_neighbors = kg[entity_id]
        n_neighbor = len(all_neighbors)
        if n_neighbor == 0:
            logger.info(f'No neighbor for {entity_id}')
            np_neighbor_ctr += 1
            continue
        sample_neighbors = np.random.choice(
            n_neighbor,
            neighbor_sample_size,
            replace=False if n_neighbor >= neighbor_sample_size else True
        )

        adj_entity[entity_id] = np.array([all_neighbors[neigh_entity][0] for neigh_entity in sample_neighbors])
        adj_relation[entity_id] = np.array([all_neighbors[neigh_entity][1] for neigh_entity in sample_neighbors])

    return adj_entity, adj_relation

def read_feature(file_path: str, entity2id: dict, drug_vocab: dict,dataset:str):
    '''
    Reads a file containing drug structure feature vectors (derived after 
    onehot encoding and pca smiles splitted word) and returns a numpy array.
    Args:
        file_path (str): Path to the drug structure feature file.
        entity2id (dict): Dictionary mapping entity names to new identifiers.
        drug_vocab (dict): Dictionary mapping drug names to new identifiers.
        dataset (str): Dataset name.
    Returns:
        drug_smiles (np.array): Array of drug structure feature vectors.
    '''
    drug_smiles = np.zeros(shape=(len(drug_vocab), 64), dtype=float)
    pca_smiles_df = pd.read_csv(file_path, sep=',', on_bad_lines='skip')
    data = pca_smiles_df.values.tolist()
    i, j = 0, 0
    for drug in data:
        drug_name = ''
        if dataset == 'kegg':
            id = drug[-1]       #kegg -1 pdd -2
            drug_name = '<http://bio2rdf.org/kegg:' + id + '>'
        elif dataset == 'pdd':
            id = drug[-2]
            drug_name = 'http://bio2rdf.org/drugbank:' + id
        if drug_name not in entity2id:
            j += 1
        else:
            i += 1
            drug_feature = drug[0:64]
            drug_smiles[entity2id[drug_name]] = np.array(drug_feature)
    
    logger.info(f'{i} drugs have smiles feature, {j} drugs do not have smiles feature')
    return drug_smiles

def read_sim(file_path: str, entity2id: dict, drug_vocab: dict):
    # 从文件中读取到dic中
    # 将drug名称转换为id编号，新dic
    # 生成np.matrix，返回
    # 后续处理在model中完成
    drug_sim = np.zeros(shape=(len(entity2id), len(entity2id)), dtype='float32')  # float64占用内存太大，改为32
    data = pd.read_csv(file_path, header=0, index_col=0)
    data = data.to_dict()
    i = 0
    for drug1 in data:
        drug_name1 = '<http://bio2rdf.org/kegg:' + drug1 + '>'
        if drug_name1 in entity2id:
            for drug2 in data[drug1]:
                drug_name2 = '<http://bio2rdf.org/kegg:' + drug2 + '>'
                if drug_name2 in entity2id:
                    i += 1
                    drug_sim[entity2id[drug_name1]][entity2id[drug_name2]] = data[drug1][drug2]
    print(i, len(drug_vocab) * len(drug_vocab))
    return drug_sim

def process_data(dataset: str, neighbor_sample_size: int, K: int):
    '''
    Method to process the raw data into the format required by the model and 
    train KGNN model using K Fold Cross Validation

    Args:
        dataset (str): The name of dataset to process [kegg, pdd]
        neighbor_sample_size (int): The number of neighbors to sample for each node
        K (int): The number of Cross Validation Folds
    '''
    # Create empty dictionaries for drug, entity, and relation vocabulary
    drug_vocab = {}
    entity_vocab = {}
    relation_vocab = {}

    # Read entity2id file and get entity2id dictionary
    entity2id = read_entity2id_file(config.ENTITY2ID_FILE[dataset], drug_vocab, entity_vocab, dataset)
    
    # Save drug and entity vocabulary as pickled files
    logger.info(f'Saving drug and entity vocabulary as pickled files')
    pickle_dump(format_filename(config.PROCESSED_DATA_DIR, config.DRUG_VOCAB_TEMPLATE, dataset=dataset), drug_vocab)
    pickle_dump(format_filename(config.PROCESSED_DATA_DIR, config.ENTITY_VOCAB_TEMPLATE, dataset=dataset), entity_vocab)

    # Read approved example file, convert to numpy array, and save as npy file
    logger.info(f'Reading approved example file, converting to numpy array, and saving as npy file Interaction Matrix')
    approved_example_file = format_filename(config.PROCESSED_DATA_DIR, config.DRUG_EXAMPLE, dataset=dataset)
    approved_examples = read_approved_example_file(config.EXAMPLE_FILE[dataset], config.SEPARATOR[dataset], drug_vocab)
    np.save(approved_example_file, approved_examples)
    logger.info(f'Saved approved example file as {approved_example_file}')

    # Processed npy absolute file paths
    adj_entity_file = format_filename(config.PROCESSED_DATA_DIR, config.ADJ_ENTITY_TEMPLATE, dataset=dataset)
    adj_relation_file = format_filename(config.PROCESSED_DATA_DIR, config.ADJ_RELATION_TEMPLATE, dataset=dataset)
    drug_vocab_npy = format_filename(config.PROCESSED_DATA_DIR, config.DRUG_VOCAB_TEMPLATE, dataset=dataset)
    entity_vocab_npy = format_filename(config.PROCESSED_DATA_DIR, config.ENTITY_VOCAB_TEMPLATE, dataset=dataset)
    relation_vocab_npy = format_filename(config.PROCESSED_DATA_DIR, config.RELATION_VOCAB_TEMPLATE, dataset=dataset)

    all_files = [adj_entity_file, adj_relation_file, drug_vocab_npy, entity_vocab_npy, relation_vocab_npy]

    if not all([os.path.exists(file_) for file_ in all_files]):
        # Read KG file and get adjacency entities and relations and also create relation vocabulary
        adj_entity, adj_relation = read_kg(config.KG_FILE[dataset], entity_vocab, relation_vocab,
                                        neighbor_sample_size)
        logger.info(f'Saving adjacency Entities as npy files at {adj_entity_file}')
        np.save(adj_entity_file, adj_entity)  
        logging.info(f'Saving adjacency Relation as npy files at {adj_relation_file}')
        np.save(adj_relation_file, adj_relation)
        
        logger.info(f'Saving drug, entity, and relation vocabulary as npy files')
        pickle_dump(drug_vocab_npy, drug_vocab)
        pickle_dump(entity_vocab_npy, entity_vocab)
        pickle_dump(relation_vocab_npy, relation_vocab)
    
    '''
    # drug_sim = read_sim(os.path.join(os.getcwd()+'/raw_data'+'/kegg/kegg_sim.csv'),entity2id,drug_vocab)
    # drug_sim_file = format_filename(PROCESSED_DATA_DIR, DRUG_SIM_TEMPLATE, dataset=dataset)
    # np.save(drug_sim_file, drug_sim,allow_pickle=True)
    # print('Logging Info - Saved:', drug_sim_file)

    # adj = drug_sim
    # adj = sp.csr_matrix(adj)
    # sp.save_npz(drug_sim_file, adj)  
    '''

    # Read drug feature from file, convert to numpy array, and save as npy file
    logger.info(f'Reading drug feature from file, converting to numpy array, and saving as npy file')
    drug_feature = read_feature(os.path.join(config.PROCESSED_DATA_DIR + '/pca_smiles_kegg.csv'), entity2id, drug_vocab,dataset)
    drug_feature_file = format_filename(config.PROCESSED_DATA_DIR, config.DRUG_FEATURE_TEMPLATE, dataset=dataset)
    np.save(drug_feature_file, drug_feature, allow_pickle=True)
    logger.info(f'Saved drug feature npy: {drug_feature_file}')
    
    # Start cv training
    logger.info(f'Starting {K} fold cross validation training for {dataset} dataset')
    cross_validation(K, approved_examples, dataset, neighbor_sample_size)

def cross_validation(K_fold, train_examples, dataset, neighbor_sample_size):
    '''
    Method to perform K Fold Cross Validation on the dataset
    
    Args:
        K_fold (int): The number of Cross Validation Folds
        train_examples (list): The list of train_examples [drug_id, entity_id, label]
        dataset (str): The name of dataset to process [kegg, pdd]
        neighbor_sample_size (int): The number of neighbors to sample for each node

    Returns:
        None # Train and save the model and logs
    '''
    # Create a dictionary to store the indices of the examples in each fold
    subsets_folds = dict()
    # Number of train_examples in each fold
    n_subsets = int(len(train_examples) / K_fold)
    # Set of indices of train_examples that are not yet assigned to a fold
    remain = set(range(0, len(train_examples) - 1))

    logger.info(f'Creating {K_fold} - fold using total train examples: {len(train_examples)}')
    # Assign indices to each K fold
    for i in reversed(range(0, K_fold - 1)):
        subsets_folds[i] = random.sample(remain, n_subsets)
        remain = remain.difference(subsets_folds[i])
    
    # Assign the remaining indices to the last fold
    subsets_folds[K_fold - 1] = remain
    # List of all aggregator types to use for experimentations
    aggregator_types = ['sum', 'concat', 'neigh', 'average']
    
    logger.info(f'Running {K_fold} fold cross validation for {dataset} dataset')
    # Create a dictionary to store the results of each fold
    for agg_method in aggregator_types:
        count = 1
        agg_method_results = {'dataset': dataset, 'aggregator_type': agg_method, \
                              'avg_auc': 0.0, 'avg_acc': 0.0, 'avg_f1': 0.0, 'avg_aupr': 0.0}
        # Create train and test for each fold 
        for fold in reversed(range(0, K_fold)):
            logger.info(f'Creating Train, Test and Val for  agg method : {agg_method} fold: {fold} ')
            # using fold as test and split to generate validation
            test_d = train_examples[list(subsets_folds[fold])]
            val_data, test_data = train_test_split(test_d, test_size=0.5)
            # Create train data by combining all other folds
            train_d = []
            for fold_ in range(0, K_fold):
                if fold != fold_:
                    train_d.extend(train_examples[list(subsets_folds[fold_])])
            train_data = np.array(train_d)
            # creating simple experiment name for loggig using callback and tensorboard
            exp_name_simple = 'tf_logs/exp_{}/{}_{}CV_{}'.format(dataset, agg_method, fold, datetime.now().strftime('%Y%m%d_%H%M%S'))
            logger.info(f'Start Training for {dataset} dataset with {agg_method} aggregator for'
                            f'fold {fold} with experiment name {exp_name_simple}')
            
            if dataset == 'kegg':
                train_log = train(
                    kfold=count,
                    dataset=dataset,
                    train_d=train_data,
                    dev_d=val_data,
                    test_d=test_data,
                    neighbor_sample_size=neighbor_sample_size,
                    embed_dim=32,
                    n_depth=2,
                    l2_weight=1e-7,
                    lr=2e-2,
                    optimizer_type='adam',
                    batch_size=2048,
                    aggregator_type=agg_method,
                    n_epoch=50, 
                    exp_name_simple=exp_name_simple,
                    callbacks_to_add=['modelcheckpoint', 'earlystopping'],
                )
            elif dataset == 'pdd':
                train_log = train(
                    kfold=count,
                    dataset=dataset,
                    train_d=train_data,
                    dev_d=val_data,
                    test_d=test_data,
                    neighbor_sample_size=neighbor_sample_size,
                    embed_dim=64,
                    n_depth=2,
                    l2_weight=1e-7,
                    lr=1e-2,
                    optimizer_type='adam',
                    batch_size=1024,
                    aggregator_type=agg_method,
                    n_epoch=50,
                    exp_name_simple=exp_name_simple,
                    callbacks_to_add=['modelcheckpoint', 'earlystopping'],
                )
            count += 1
            agg_method_results['avg_auc'] = agg_method_results['avg_auc'] + train_log['test_auc']
            agg_method_results['avg_acc'] = agg_method_results['avg_acc'] + train_log['test_acc']
            agg_method_results['avg_f1'] = agg_method_results['avg_f1'] + train_log['test_f1']
            agg_method_results['avg_aupr'] = agg_method_results['avg_aupr'] + train_log['test_aupr']
        
        # Calculate the average of each metric for all folds
        for key in agg_method_results:
            if key == 'aggregator_type' or key == 'dataset':
                continue
            agg_method_results[key] = agg_method_results[key] / K_fold
        # Write the results to the log file
        write_log(format_filename(config.LOG_DIR, config.RESULT_LOG[dataset]), agg_method_results, 'a')
        logger.info('#'*100)
        logger.info(f'{K_fold} fold result: avg_auc: {agg_method_results["avg_auc"]}, avg_acc: {agg_method_results["avg_acc"]}, avg_f1: {agg_method_results["avg_f1"]}, avg_aupr: {agg_method_results["avg_aupr"]}')


if __name__ == '__main__':
    logger.info('Setting up all required directories')
    check_folder_exists = [config.PROCESSED_DATA_DIR, config.LOG_DIR, config.MODEL_SAVED_DIR]
    for folder in check_folder_exists:
        if not os.path.exists(folder):
            logger.info(f'Creating {folder}')
            os.makedirs(folder)

    model_config = config.ModelConfig()
    #process_data('kegg', config.NEIGHBOR_SIZE['kegg'], 5)
    process_data('pdd', config.NEIGHBOR_SIZE['pdd'], 5)



