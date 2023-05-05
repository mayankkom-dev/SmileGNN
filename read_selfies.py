import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import logging, config
import sys, os
import selfies
import swifter

logging.basicConfig(level=logging.INFO, format= '[%(asctime)s] [%(pathname)s:%(lineno)d] [%(levelname)s] - %(message)s',
     datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('log/run.log')]) 
logger = logging.getLogger(__name__)


def smi2id(smiles,vocalbulary):
    sequence_id=[]
    for i in range(len(smiles)):
        smi_id=[]
        for j in range(len(smiles[i])):
            smi_id.append(vocalbulary.index(smiles[i][j]))
        sequence_id.append(smi_id)
    return sequence_id

def one_hot_encoding(smi, vocalbulary):
    res=[]
    for i in vocalbulary:
        if i in smi:
            res.append(1)
        else:
            res.append(0)
    return res

def encode_selfies(p_data, output_file):
    '''
    To encode selfies, we need to first create an alphabet of all the possible symbols
    that can be found in the selfies. Then, we can use the selfies library to encode
    the selfies into a one-hot encoding.

    Args:
        p_data (pandas.DataFrame): The dataframe containing the selfies to be encoded.
        output_file (str): The path to the output file to save the encoded selfies.

    Returns:
        pandas.DataFrame: The dataframe containing the encoded selfies.
    '''
    alphabet = selfies.get_alphabet_from_selfies(p_data['selfies'].values.tolist())
    alphabet.add('[nop]')
    alphabet = list(sorted(alphabet))
    pad_to_len = max(selfies.len_selfies(s) for s in p_data['selfies'].values.tolist())  # 5
    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
    # explode columns here 
    # encoded_selfies = p_data['selfies'].swifter.apply(lambda selfie: \
    #                                     np.array(selfies.selfies_to_encoding(
    #                                         selfie, vocab_stoi=symbol_to_idx, 
    #                                         pad_to_len=pad_to_len, enc_type="one_hot"
    #                                         )).flatten().tolist())
    encoded_selfies = p_data['selfies'].swifter.apply(lambda selfie: \
                                        one_hot_encoding(
                                            selfie, list(symbol_to_idx.keys()), 
                                        ))
    
    df = p_data[['DrugBank ID', 'KEGG Drug ID']].rename(columns={'DrugBank ID': 'dbid', 'KEGG Drug ID': 'keggid'})
    df['selfies'] = encoded_selfies.values.tolist()
    # df = pd.DataFrame(encoded_selfies.values.tolist())
    # print(df.shape)
    # df['dbid'] = p_data['DrugBank ID'].values.tolist()
    # df['keggid'] = p_data['KEGG Drug ID'].values.tolist()
    
    new_df = df.dropna(axis=0, subset=['keggid'])
    logger.info(f'Writing encoded selfies to {output_file}')
    # new_df.to_csv(output_file, encoding='utf-8',index=False)
    return new_df

def calculate_pca(profile_file, output_file, p_data, embed_dim):
    '''
    Calculate the PCA of the encoded selfies.

    Args:
        profile_file [str | pandas.DataFrame]: The path to the file containing the encoded selfies.
        output_file (str): The path to the output file to save the PCA of the encoded selfies.
        p_data (pandas.DataFrame): The dataframe containing the encoded selfies.

    Returns:
        pandas.DataFrame: The dataframe containing the PCA of the encoded selfies.
    '''
    pca = PCA(copy=True, iterated_power='auto', n_components=embed_dim,
              svd_solver='auto', tol=0.0, whiten=False)
    if isinstance(profile_file, str):
        df = pd.read_csv(profile_file) #, index_col=0
    else:
        df = profile_file 

    X = np.array(df[df.columns[-1]].tolist()) # dbid, keggid drop
    X = pca.fit_transform(X)

    new_df = pd.DataFrame(X, columns=['PC_%d' % (i + 1) for i in range(embed_dim)], index=df.index)
    print(new_df.shape)
    new_df['dbid'] = df['dbid'].values.tolist()
    new_df['keggid'] = df['keggid'].values.tolist()
    print(new_df.head())
    new_df = new_df.dropna(axis=0, subset=['keggid'])
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    new_df.to_csv(output_file, encoding='utf-8',index=False)
    return new_df

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_everything(42)
    drugbank_smiles_data_loc = f'{config.RAW_DATA_DIR}/drugbank_all_structure_links.csv'
    if os.path.exists(f'{config.PROCESSED_DATA_DIR}/selfies/selfies_all.csv'):
        logger.info(f'Loading already processed selfies from {config.PROCESSED_DATA_DIR}/selfies/selfies_all.csv')
        db_selfies_df = pd.read_csv(f'{config.PROCESSED_DATA_DIR}/selfies/selfies_all.csv') 
    else:              
        logger.info(f'Reading drunbank all smiles data csv files at {drugbank_smiles_data_loc}')
        db_smiles_df = pd.read_csv(drugbank_smiles_data_loc, encoding="Windows-1256")
        db_smiles_df = db_smiles_df.dropna(axis=0,subset=["SMILES"])
        print(db_smiles_df.columns.values.tolist())
        print(db_smiles_df.shape)
        all_smiles = db_smiles_df['SMILES'].values.tolist()
        logger.info(f'Converting all Smiles representation to SELFIES')
        all_selfies = []
        err_smiles_ct = 0
        for smiles in all_smiles:
            try:
                selfie_rep = selfies.encoder(smiles)
                if '.' in selfie_rep:
                    selfie_rep = None
                    err_smiles_ct += 1 
            except selfies.EncoderError as err:
                selfie_rep = None
                err_smiles_ct += 1
            all_selfies.append(selfie_rep)
        
        logger.info(f'Number of smiles that could not be converted to SELFIES: {err_smiles_ct}')
        db_smiles_df['selfies'] = all_selfies
        db_selfies_df = db_smiles_df.dropna(axis=0,subset=["selfies"]).copy(deep=True)
        os.makedirs(f"{config.PROCESSED_DATA_DIR}/selfies", exist_ok=True)
        db_selfies_df.to_csv(f"{config.PROCESSED_DATA_DIR}/selfies/selfies_all.csv", encoding='utf-8',index=False)
        logger.info(f'Saved processed selfies at {config.PROCESSED_DATA_DIR}/selfies/selfies_all.csv')

    logger.info(f'OneHotEncoding selfies')
    input_file = f"{config.PROCESSED_DATA_DIR}/selfies/encoded_selfies_all.csv"
    encoded_df = encode_selfies(db_selfies_df, input_file)
    embed_dim = 64
    output_file = f"{config.PROCESSED_DATA_DIR}/selfies/pca_selfies_kegg_{embed_dim}.csv"
    logger.info(f'Calculating PCA and saving at {output_file}')
    new_data = calculate_pca(encoded_df, output_file, db_selfies_df, embed_dim)

