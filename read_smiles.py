import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import logging, config
import sys

logging.basicConfig(level=logging.INFO, format= '[%(asctime)s] [%(pathname)s:%(lineno)d] [%(levelname)s] - %(message)s',
     datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('log/run.log')]) 
logger = logging.getLogger(__name__)

def smi_preprocessing(smi_sequence):
    splited_smis=[]
    length=[]
    end="/n"
    begin="&"
    element_table=["C","N","B","O","P","S","F","Cl","Br","I","(",")","=","#"]
    for i in range(len(smi_sequence)):
        smi=smi_sequence[i]
        splited_smi=[]
        j=0
        while j<len(smi):
            smi_words=[]
            if smi[j]=="[":
                smi_words.append(smi[j])
                j=j+1
                while smi[j]!="]":
                    smi_words.append(smi[j])
                    j=j+1
                smi_words.append(smi[j])
                words = ''.join(smi_words)
                splited_smi.append(words)
                j=j+1

            else:
                smi_words.append(smi[j])

                if j+1<len(smi[j]):
                    smi_words.append(smi[j+1])
                    words = ''.join(smi_words)
                else:
                    smi_words.insert(0,smi[j-1])
                    words = ''.join(smi_words)

                if words not in element_table:
                    splited_smi.append(smi[j])
                    j=j+1
                else:
                    splited_smi.append(words)
                    j=j+2

        splited_smi.append(end)
        splited_smi.insert(0,begin)
        splited_smis.append(splited_smi)
    return splited_smis

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

def encode_smiles(smi,vocalbulary, p_data, output_file):
    res = []
    for drug_smile in smi:
        res.append(one_hot_encoding(drug_smile,vocalbulary))
    print(len(res))
    df = pd.DataFrame(res)
    print(df.shape)
    df['dbid'] = p_data['DrugBank ID'].values.tolist()
    df['keggid'] = p_data['KEGG Drug ID'].values.tolist()
    new_df = df.dropna(axis=0, subset=['keggid'])
    new_df.to_csv(output_file, encoding='utf-8',index=False)

def calculate_pca(profile_file, output_file, p_data):
    pca = PCA(copy=True, iterated_power='auto', n_components=96, random_state=None,
              svd_solver='auto', tol=0.0, whiten=False)
    df = pd.read_csv(profile_file) #, index_col=0

    X = df[df.columns[:-2]].values # dbid, keggid drop
    X = pca.fit_transform(X)

    new_df = pd.DataFrame(X, columns=['PC_%d' % (i + 1) for i in range(96)], index=df.index)
    print(new_df.shape)
    new_df['dbid'] = df['dbid'].values.tolist()
    new_df['keggid'] = df['keggid'].values.tolist()
    print(new_df.head())
    # new_df = new_df.dropna(axis=0, subset=['keggid'])
    new_df.to_csv(output_file, encoding='utf-8',index=False)
    return new_df



if __name__ == '__main__':
    drugbank_smiles_data_loc = f'{config.RAW_DATA_DIR}/drugbank_all_structure_links.csv'
    logger.info(f'Reading drunbank all smiles data csv files at {drugbank_smiles_data_loc}')
    db_smiles_df = pd.read_csv(drugbank_smiles_data_loc, encoding="Windows-1256")
    db_smiles_df = db_smiles_df.dropna(axis=0,subset=["SMILES"])
    print(db_smiles_df.columns.values.tolist())
    print(db_smiles_df.shape)
    all_smiles = db_smiles_df['SMILES'].values.tolist()
    logger.info(f'Pre-processing smiles to generate splitted words')
    smi = smi_preprocessing(all_smiles)
    print(smi[0][:5])
    logger.info(f'Computing vocabulary')
    vocalbulary=[splitted_word for splitted_word_list in smi for splitted_word in splitted_word_list]
    vocalbulary=list(set(vocalbulary))
    print(vocalbulary, len(vocalbulary))
    docs = dict(zip(vocalbulary, range(len(vocalbulary))))
    print(docs)
    
    logger.info(f'OneHotEncoding smiles')
    input_file = f"{config.PROCESSED_DATA_DIR}/smiles/encoded_smiles_all.csv"
    print(one_hot_encoding(smi[0], vocalbulary))
    encode_smiles(smi,vocalbulary, db_smiles_df, input_file)
    
    output_file = f"{config.PROCESSED_DATA_DIR}/smiles/pca_smiles_kegg.csv"
    logger.info(f'Calculating PCA and saving at {output_file}')
    new_data = calculate_pca(input_file, output_file, db_smiles_df)

