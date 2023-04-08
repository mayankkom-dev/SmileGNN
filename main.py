import os
import gc
import time
import tensorflow as tf
import scipy.sparse as sp
# from SmileGNN.layers.feature import *
import pandas as pd
import time
import numpy as np
from collections import defaultdict
from keras import backend as K
from keras import optimizers
from tensorflow.keras.callbacks import TensorBoard
from utils import load_data, pickle_load, format_filename, write_log
from models import KGCN
from config import ModelConfig, PROCESSED_DATA_DIR,  ENTITY_VOCAB_TEMPLATE, \
    RELATION_VOCAB_TEMPLATE, ADJ_ENTITY_TEMPLATE, ADJ_RELATION_TEMPLATE, LOG_DIR, PERFORMANCE_LOG, \
    DRUG_VOCAB_TEMPLATE, DRUG_FEATURE_TEMPLATE,DRUG_SIM_TEMPLATE, RESULT_DATA_DIR
import logging, sys

logging.basicConfig(level=logging.INFO, format= '[%(asctime)s] [%(pathname)s:%(lineno)d] [%(levelname)s] - %(message)s',
     datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('log/run.log')]) 
logger = logging.getLogger(__name__)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_optimizer(op_type, learning_rate):
    if op_type == 'sgd':
        return optimizers.SGD(learning_rate)
    elif op_type == 'rmsprop':
        return optimizers.RMSprop(learning_rate)
    elif op_type == 'adagrad':
        return optimizers.Adagrad(learning_rate)
    elif op_type == 'adadelta':
        return optimizers.Adadelta(learning_rate)
    elif op_type == 'adam':
        return optimizers.Adam(learning_rate, clipnorm=6)
    else:
        raise ValueError('Optimizer Not Understood: {}'.format(op_type))

def compare_y(drug1, drug2, y_train, y_pred, y_pred_2, outfile):
    y_train = y_train.flatten()
    new_DDI = []
    for i in range(len(y_train)):
        if y_pred_2[i] == 1 and y_train[i] == 0:
            new_DDI.append([drug1[i],drug2[i],y_pred[i]])
    df = pd.DataFrame(new_DDI, columns=['drug1','drug2','score'])
    df.to_csv(outfile, index=False, encoding='utf-8')
    logger.info(f"Found {df.shape[0]} new DDI dumping at {outfile}")
    time.sleep(10)


def train(train_d, dev_d, test_d, kfold, dataset, neighbor_sample_size, embed_dim, n_depth, l2_weight, lr, optimizer_type,
          batch_size, aggregator_type, n_epoch, exp_name_simple, callbacks_to_add=None, overwrite=True):
    # setting up model config using param provided
    config = ModelConfig()
    config.neighbor_sample_size = neighbor_sample_size
    config.embed_dim = embed_dim
    config.n_depth = n_depth
    config.l2_weight = l2_weight
    config.dataset = dataset
    config.K_Fold = kfold
    config.lr = lr
    config.optimizer = get_optimizer(optimizer_type, lr)
    config.batch_size = batch_size
    config.aggregator_type = aggregator_type
    config.n_epoch = n_epoch
    config.callbacks_to_add = callbacks_to_add

    config.drug_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                             DRUG_VOCAB_TEMPLATE,
                                                             dataset=dataset)))
    config.entity_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                               ENTITY_VOCAB_TEMPLATE,
                                                               dataset=dataset)))
    config.relation_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                                 RELATION_VOCAB_TEMPLATE,
                                                                 dataset=dataset)))
    config.adj_entity = np.load(format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE,
                                                dataset=dataset))
    config.adj_relation = np.load(format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE,
                                                  dataset=dataset))

    config.drug_feature = np.load(format_filename(PROCESSED_DATA_DIR, DRUG_FEATURE_TEMPLATE, dataset=dataset),allow_pickle=True)
    config.callbacks_tboard = TensorBoard(log_dir=exp_name_simple,
                                        histogram_freq=0,
                                        write_graph=True,
                                        write_images=False,
                                        update_freq='batch',
                                        profile_batch=0,
                                        embeddings_freq=1)


    config.exp_name = f'kgcn_{dataset}_neigh_{neighbor_sample_size}_embed_{embed_dim}_depth_' \
                      f'{n_depth}_agg_{aggregator_type}_optimizer_{optimizer_type}_lr_{lr}_' \
                      f'batch_size_{batch_size}_epoch_{n_epoch}'
    callback_str = '_' + '_'.join(config.callbacks_to_add)
    callback_str = callback_str.replace('_modelcheckpoint', '').replace('_earlystopping', '')
    config.exp_name += callback_str

    # dict to log training params and result
    train_log = {'exp_name': config.exp_name, 'batch_size': batch_size, 'optimizer': optimizer_type,
                 'epoch': n_epoch, 'learning_rate': lr}
    logger.info(f'Starting Experiment: {config.exp_name}')
    model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    model = KGCN(config)

    train_data=np.array(train_d)
    valid_data=np.array(dev_d)
    test_data=np.array(test_d)

    if not os.path.exists(model_save_path) or overwrite:
        start_time = time.time()
        model.fit(x_train=[train_data[:, :1], train_data[:, 1:2]], y_train=train_data[:, 2:3],
                  x_valid=[valid_data[:, :1], valid_data[:, 1:2]], y_valid=valid_data[:, 2:3])
        elapsed_time = time.time() - start_time
        logger.info('Training time: %s' % time.strftime("%H:%M:%S",
                                                                 time.gmtime(elapsed_time)))
        train_log['train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    logger.info('Evaluating over valid data:')
    model.load_best_model()
    y_pred, y_pred_2, auc, acc, f1, aupr = model.score(x=[valid_data[:, :1], valid_data[:, 1:2]], y=valid_data[:, 2:3])

    logger.info(f'Train Evaluation Metrics: val_auc: {auc}, val_acc: {acc}, val_f1: {f1}, val_aupr: {aupr}')
    train_log['val_auc'] = auc
    train_log['val_acc'] = acc
    train_log['val_f1'] = f1
    train_log['val_aupr']=aupr
    train_log['k_fold']=kfold
    train_log['dataset']=dataset
    train_log['aggregate_type']=config.aggregator_type
    
    if 'swa' in config.callbacks_to_add:
        model.load_swa_model()
        print('Logging Info - Evaluate over valid data based on swa model:')
        y_pred,y_pred_2,auc, acc, f1,aupr = model.score(x=[valid_data[:, :1], valid_data[:, 1:2]], y=valid_data[:, 2:3])

        train_log['swa_val_auc'] = auc
        train_log['swa_val_acc'] = acc
        train_log['swa_val_f1'] = f1
        train_log['swa_val_aupr']=aupr
        print(f'Logging Info - swa_val_auc: {auc}, swa_val_acc: {acc}, swa_val_f1: {f1}, swa_val_aupr: {aupr}')
    
    logger.info('Evaluate over test data:')
    model.load_best_model()
    y_pred,y_pred_2,auc, acc, f1, aupr = model.score(x=[test_data[:, :1], test_data[:, 1:2]], y=test_data[:, 2:3])
    
    train_log['test_auc'] = auc
    train_log['test_acc'] = acc
    train_log['test_f1'] = f1
    train_log['test_aupr'] =aupr
    
    logger.info('Finding new DDI Interaction')
    outfile = f"{RESULT_DATA_DIR}/pdd_new_DDI_pdd_score.csv"
    compare_y(test_data[:, :1], test_data[:, 1:2], test_data[:, 2:3], y_pred, y_pred_2, outfile)
    
    logger.info(f'Test Evaluation metrics - test_auc: {auc}, test_acc: {acc}, test_f1: {f1}, test_aupr: {aupr}')
    if 'swa' in config.callbacks_to_add:
        model.load_swa_model()
        logger.info('Evaluate over test data based on swa model:')
        y_pred,y_pred_2,auc, acc, f1,aupr = model.score(x=[test_data[:, :1], test_data[:, 1:2]], y=test_data[:, 2:3])
        train_log['swa_test_auc'] = auc
        train_log['swa_test_acc'] = acc
        train_log['swa_test_f1'] = f1
        train_log['swa_test_aupr'] = aupr
        logger.info(f'swa_test_auc: {auc}, swa_test_acc: {acc}, swa_test_f1: {f1}, swa_test_aupr: {aupr}')
    
    train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG), log=train_log, mode='a')
    del model
    gc.collect()
    K.clear_session()
    return train_log

