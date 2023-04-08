import os
import tensorflow as tf

RAW_DATA_DIR = os.getcwd()+'/raw_data'
PROCESSED_DATA_DIR = os.getcwd()+'/processed_data'
RESULT_DATA_DIR = os.getcwd()+'/result_data'
LOG_DIR = os.getcwd()+'/log'
MODEL_SAVED_DIR = os.getcwd()+'/ckpt'

KG_FILE = {
           'drugbank':os.path.join(RAW_DATA_DIR,'drugbank','train2id.txt'),
           'kegg':os.path.join(RAW_DATA_DIR,'kegg','train2id.txt'),
           'pdd':os.path.join(RAW_DATA_DIR,'pdd','train2id.txt')}
ENTITY2ID_FILE = {
                    'drugbank':os.path.join(RAW_DATA_DIR,'drugbank','entity2id.txt'),
                    'kegg':os.path.join(RAW_DATA_DIR,'kegg','entity2id.txt'),
                    'pdd':os.path.join(RAW_DATA_DIR,'pdd','entity2id.txt')}
EXAMPLE_FILE = {
               'drugbank':os.path.join(RAW_DATA_DIR,'drugbank','approved_example.txt'),
               'kegg':os.path.join(RAW_DATA_DIR,'kegg','approved_example.txt'),
               'pdd':os.path.join(RAW_DATA_DIR,'pdd','approved_example.txt')}
SEPARATOR = {'drug':'\t','kegg':'\t','pdd':' '}
THRESHOLD = {'drug':4,'kegg':4,'pdd':4} #添加drug修改
NEIGHBOR_SIZE = {'drug':4,'kegg':4,'pdd':4}

#
DRUG_VOCAB_TEMPLATE = '{dataset}_drug_vocab.pkl'
ENTITY_VOCAB_TEMPLATE = '{dataset}_entity_vocab.pkl'
RELATION_VOCAB_TEMPLATE = '{dataset}_relation_vocab.pkl'
ADJ_ENTITY_TEMPLATE = '{dataset}_adj_entity.npy'
ADJ_RELATION_TEMPLATE = '{dataset}_adj_relation.npy'
TRAIN_DATA_TEMPLATE = '{dataset}_train.npy'
DEV_DATA_TEMPLATE = '{dataset}_dev.npy'
TEST_DATA_TEMPLATE = '{dataset}_test.npy'
DRUG_FEATURE_TEMPLATE = '{dataset}_drug_feature.npy'
DRUG_SIM_TEMPLATE = '{dataset}_drug_sim.npz'
#RESULT_LOG='result.txt'
RESULT_LOG={'drugbank':'drugbank_result_test_avg_k_fold.txt','kegg':'kegg_result_test_avg_k_fold.txt','pdd':'pdd_result_test_avg_k_fold.txt'}
PERFORMANCE_LOG = 'SmileGNN_performance_per_fold.log'
DRUG_EXAMPLE='{dataset}_examples.npy'
EPOCH_END_LOG = 'train_epoch_end_log.txt'

class ModelConfig(object):
    def __init__(self):
        self.neighbor_sample_size = 4 # neighbor sampling size
        self.embed_dim = 32  # dimension of embedding
        self.n_depth = 2    # depth of receptive field
        self.l2_weight = 1e-7  # l2 regularizer weight
        self.lr = 2e-2  # learning rate
        self.batch_size = 65536
        self.aggregator_type = 'sum'
        self.n_epoch = 50
        self.optimizer = 'adam'

        self.drug_vocab_size = None
        self.entity_vocab_size = None
        self.relation_vocab_size = None
        self.adj_entity = None
        self.adj_relation = None
        self.drug_feature = None
        self.drug_sim = None

        self.exp_name = None
        self.model_name = None

        self.att = None
        self.middle_att = None

        # checkpoint configuration 设置检查点
        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_auc'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stoping configuration
        self.early_stopping_monitor = 'val_auc'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1
        self.dataset='drug'
        self.K_Fold=1
        self.callbacks_to_add = None

        # config for learning rating scheduler and ensembler
        self.swa_start = 3
