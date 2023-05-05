# -*- coding: utf-8 -*-
from keras.layers import *
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K  # use computable function
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve
import sklearn.metrics as m
from layers import Aggregator
from callbacks import KGCNMetric
from models.base_model import BaseModel
# from tensorflow.keras.layers import Layer
# from SmileGNN.layers.feature import *
# from tensorflow import sparse_tensor_dense_matmul
from tensorflow.sparse import sparse_dense_matmul
from tensorflow.keras.regularizers import l2
import tensorflow as tf


class TransformerUnit(tf.keras.layers.Layer):
    def __init__(self, config):
        super(TransformerUnit, self).__init__()
        self.config = config
        self.input_size = config.embed_dim
        self.num_attn_heads = config.num_heads
        self.mlp_embed_factor = config.mlp_embed_factor
        self.nonlin_func = config.nonlin_func
        self.pdropout = config.dropout

    def build(self, input_shape):
        self.attn = MultiHeadAttention(self.config, num_heads=self.num_attn_heads,
                                        key_dim=self.input_size, 
                                        value_dim=self.input_size)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_layer = tf.keras.layers.Dropout(self.pdropout)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.input_size * self.mlp_embed_factor, activation=self.nonlin_func, kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(units=self.input_size, kernel_initializer='glorot_uniform')
        ])
        super().build(input_shape)
    
    def call(self, query, key, value, mask=None):
        attn_output = self.attn(query, key, value, mask)
        out1 = self.layernorm1(attn_output + key)
        out1 = self.dropout_layer(out1)         
        mlp_output = self.mlp(out1)
        out2 = self.layernorm2(out1 + mlp_output)
        out2 = self.dropout_layer(out2)
        
        return out2

    def get_config(self):
        config = super(TransformerUnit, self).get_config()
        config.update({"config": self.config})
        return config

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, config, num_heads, key_dim, value_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.config = config
        # Initialize trainable weights for attention mechanism
        self.wq = self.add_weight(
            shape=(key_dim, num_heads * key_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='wq'
        )
        self.wk = self.add_weight(
            shape=(key_dim, num_heads * key_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='wk'
        )
        self.wv = self.add_weight(
            shape=(value_dim, num_heads * value_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='wv'
        )
        
    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(units=self.value_dim, name='dense', kernel_initializer='glorot_uniform')
        self.project = tf.keras.layers.Dense(units=self.value_dim, name='project', activation='tanh', kernel_initializer='glorot_uniform')
        
        super().build(input_shape)

    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]
        
        # Linear transformations
        q = tf.matmul(query, self.wq)
        k = tf.matmul(key, self.wk)
        v = tf.matmul(value, self.wv)
        
        # Reshape to split heads
        q = tf.reshape(q, [batch_size, -1, self.num_heads, self.key_dim])
        k = tf.reshape(k, [batch_size, -1, self.num_heads, self.key_dim])
        v = tf.reshape(v, [batch_size, -1, self.num_heads, self.value_dim])
        
        # Transpose and concatenate
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        q = self.project(q)
        k = self.project(k)
        score = q + k


        # Dot product attention
        # score = tf.matmul(q, k, transpose_b=True)
        # score = score / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        
        # Masking
        if mask is not None:
            mask = tf.expand_dims(mask, axis=1)
            score = score * mask + (1.0 - mask) * -1e9
        
        # Softmax
        attn_weights = tf.nn.softmax(score, axis=-1)
        
        # Weighted sum
        context = attn_weights * v        
        # context = tf.matmul(attn_weights, v)
        
        # Reshape
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, -1, self.num_heads * self.value_dim])
        
        output = self.dense(context)
        
        return output
    
    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({"config": self.config})
        return config

class ScoreLayer(Layer):
    def __init__(self, aggregator, activation='tanh', l2_reg=2e-6, **kwargs):
        super(ScoreLayer, self).__init__(**kwargs)
        self.aggregator = aggregator
        self.activation = activation
        self.l2_reg = l2_reg
        

    def call(self, inputs):
        embed_list = inputs
        drug1 = embed_list[0]  # [batch_size,,embed_size]
        drug2 = embed_list[1]
        drug1_f = embed_list[2]  # [batch_size,embed_size]
        drug2_f = embed_list[3]

        # aggregate drug embeddings and features
        # drug1 = self.aggregator([drug1, drug1_f])
        # drug2 = self.aggregator([drug2, drug2_f])

        # remove batch_size dimension
        drug1 = tf.squeeze(drug1, axis=1)
        drug2 = tf.squeeze(drug2, axis=1)

        # compute drug-drug score using dot product and sigmoid activation
        drug_drug_score = tf.sigmoid(tf.reduce_sum(drug1 * drug2, axis=-1, keepdims=True))

        return drug_drug_score
    
    def get_config(self):
        config = super(ScoreLayer, self).get_config()
        config.update({
            'activation': self.activation,
            'aggregator': self.aggregator,
            'l2_reg': self.l2_reg,
        })
        return config

class NeighborInfoLayer(Layer):
    def __init__(self, config, **kwargs):
        super(NeighborInfoLayer, self).__init__(**kwargs)
        self.config = config
        self.transformer_drug = TransformerUnit(config = self.config)
        self.transformer_rel = TransformerUnit(config = self.config)
        # tf keras sequential layer
        self.combine_ent_rel = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.config.embed_dim, activation= 'relu', name='combine_ent_rel', kernel_initializer=tf.keras.initializers.GlorotUniform()),
            # tf.keras.layers.Dense(units=self.config.embed_dim, activation= 'relu', name='combine_ent_rel2', kernel_initializer=tf.keras.initializers.GlorotUniform())
        ])
        # self.combine_ent_rel = tf.keras.layers.Dense(units=self.config.embed_dim, activation= 'relu', name='combine_ent_rel')
        # normalize layer
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-4, name='norm')
        # self.final_combine = tf.keras.Sequential([
        #     tf.keras.layers.Dense(units=self.config.embed_dim, activation= 'relu', name='final_combine', kernel_initializer=tf.keras.initializers.GlorotUniform()),
        #     # tf.keras.layers.Dense(units=self.config.embed_dim, activation= 'relu', name='final_combine2', kernel_initializer=tf.keras.initializers.GlorotUniform())
        # ])
        # # # Initialize trainable weights for attention mechanism
        # self.attention_drug = MultiHeadAttention(config = self.config,
        #     num_heads=self.config.num_heads,
        #     key_dim=self.config.embed_dim,
        #     value_dim=self.config.embed_dim,
        #     dropout=self.config.dropout,
        #     name="attention_drug"
        # )
        # self.attention_rel = MultiHeadAttention(config = self.config,
        #     num_heads=self.config.num_heads,
        #     key_dim=self.config.embed_dim,
        #     value_dim=self.config.embed_dim,
        #     dropout=self.config.dropout,
        #     name="attention_rel"
        # )
        self.w = self.add_weight(
            shape=(self.config.embed_dim, self.config.neighbor_sample_size),
            initializer='glorot_uniform',
            trainable=True,
            name='w_neigh'
        )
    def call(self, inputs):
        drug, rel, ent = inputs
        
        # Apply cross-attention between drug and ent
        # [batch_size, neighbor_size ** hop, embed_dim]
        # ent = self.attention_drug(query=drug, key=ent, value=ent)

        # # Apply cross-attention between drug and rel
        # # [batch_size, neighbor_size ** hop, embed_dim]
        # # rel = self.attention_rel(query=rel, key=drug, value=rel)

        # # [batch_size, neighbor_size ** hop, 1] drug-entity score
        # drug_rel_score = tf.reduce_sum(drug * rel, axis=-1, keepdims=True)
        
        # # [batch_size, neighbor_size ** hop, embed_dim]
        # weighted_ent = drug_rel_score * ent

        # # [batch_size, neighbor_size ** (hop-1), neighbor_size, embed_dim]
        # weighted_ent = tf.reshape(weighted_ent,
        #                          (tf.shape(weighted_ent)[0], -1,
        #                           self.config.neighbor_sample_size, self.config.embed_dim))

        # neighbor_embed = tf.reduce_sum(weighted_ent, axis=2)
        # return neighbor_embed
        ent = self.transformer_drug(query=drug, key=ent, value=ent)
        rel = self.transformer_rel(query=drug, key=rel, value=rel)

        # # [batch_size, neighbor_size ** hop, 1] drug-entity score
        # drug_rel_score = tf.reduce_sum(drug * rel, axis=-1, keepdims=True)
        
        # # [batch_size, neighbor_size ** hop, embed_dim]
        ent_rel = tf.concat([tf.repeat(drug, ent.shape[1], axis=1), rel, ent], axis=-1)
        weighted_ent = self.combine_ent_rel(ent_rel)
        weighted_ent = self.norm(weighted_ent)
        # weighted_ent = drug_rel_score * ent ## can be a layer
        # weighted_ent = ent * rel ## can be a layer

        # # [batch_size, neighbor_size ** (hop-1), neighbor_size, embed_dim]
        weighted_ent = tf.reshape(weighted_ent,
                                 (tf.shape(weighted_ent)[0], -1,
                                  self.config.neighbor_sample_size, self.config.embed_dim))

        k = tf.matmul(self.w, weighted_ent)
        neighbor_embed = tf.linalg.diag_part(k)

        # neighbor_embed = self.final_combine(weighted_ent) # agg layer for each neighbor aggregation
        # neighbor_embed = tf.reduce_sum(weighted_ent, axis=2)
        return neighbor_embed

    def get_config(self):
        config = super(NeighborInfoLayer, self).get_config()
        config.update({"config": self.config})
        return config
    
class ReceptiveFieldLayer(Layer):
    def __init__(self, config, **kwargs):
        super(ReceptiveFieldLayer, self).__init__(**kwargs)
        self.config = config
        self.adj_entity_matrix = tf.Variable(self.config.adj_entity, name='adj_entity', dtype=tf.int64)
        self.adj_relation_matrix = tf.Variable(self.config.adj_relation, name='adj_relation', dtype=tf.int64)
        self.n_neighbor = tf.shape(self.adj_entity_matrix)[1]

    def call(self, entity):
        neigh_ent_list = [entity]
        neigh_rel_list = []

        for i in range(self.config.n_depth):
            new_neigh_ent = tf.gather(self.adj_entity_matrix, tf.cast(
                neigh_ent_list[-1], dtype=tf.int64))
            new_neigh_rel = tf.gather(self.adj_relation_matrix, tf.cast(
                neigh_ent_list[-1], dtype=tf.int64))
            neigh_ent_list.append(
                tf.reshape(new_neigh_ent, (-1, self.n_neighbor ** (i + 1))))
            neigh_rel_list.append(
                tf.reshape(new_neigh_rel, (-1, self.n_neighbor ** (i + 1))))

        return neigh_ent_list + neigh_rel_list

    def get_config(self):
        config = super(ReceptiveFieldLayer, self).get_config()
        config.update({"config": self.config,
                       "adj_entity_matrix": self.adj_entity_matrix,
                       "adj_relation_matrix": self.adj_relation_matrix,
                       "n_neighbor": self.n_neighbor})
        return config

class N2VRandomWalkReceptiveFieldLayer(Layer):
    def __init__(self, config, **kwargs):
        super(ReceptiveFieldLayer, self).__init__(**kwargs)
        self.config = config
        self.adj_entity_matrix = tf.Variable(self.config.adj_entity, name='adj_entity', dtype=tf.int64)
        self.adj_relation_matrix = tf.Variable(self.config.adj_relation, name='adj_relation', dtype=tf.int64)
        self.n_neighbor = tf.shape(self.adj_entity_matrix)[1]

    def call(self, entity):
        neigh_ent_list = [entity]
        neigh_rel_list = []

        for i in range(self.config.n_depth):
            new_neigh_ent = tf.gather(self.adj_entity_matrix, tf.cast(
                neigh_ent_list[-1], dtype=tf.int64))
            new_neigh_rel = tf.gather(self.adj_relation_matrix, tf.cast(
                neigh_ent_list[-1], dtype=tf.int64))
            neigh_ent_list.append(
                tf.reshape(new_neigh_ent, (-1, self.n_neighbor ** (i + 1))))
            neigh_rel_list.append(
                tf.reshape(new_neigh_rel, (-1, self.n_neighbor ** (i + 1))))

        return neigh_ent_list + neigh_rel_list

    def get_config(self):
        config = super(ReceptiveFieldLayer, self).get_config()
        config.update({"config": self.config,
                       "adj_entity_matrix": self.adj_entity_matrix,
                       "adj_relation_matrix": self.adj_relation_matrix,
                       "n_neighbor": self.n_neighbor})
        return config

class FeatureLayer(Layer):
    def __init__(self, config, **kwargs):
        super(FeatureLayer, self).__init__(**kwargs)
        self.config = config
        self.drug_feature_tensor = tf.Variable(self.config.drug_feature,  name='drug_pca_feature', dtype='float32', trainable=False)
        
    def call(self, inputs):
        input_drug = inputs
        
        # [batch_size,] drug index
        drug = [input_drug][-1]
        
        # [batch_size, drug_feature_embed_dimension] drug feature tensor
        
        # [batch_size, drug_feature_embed_dimension] drug feature vector
        drug_f = tf.gather(self.drug_feature_tensor, tf.cast(drug, dtype='int32'))
        
        return drug_f

    def get_config(self):
        config = super(FeatureLayer, self).get_config()
        config.update({"config": self.config,
                       "drug_feature_tensor": self.drug_feature_tensor})
        return config
    
class KGCN(BaseModel):
    def __init__(self, config):
        super(KGCN, self).__init__(config)

    def build(self):
        input_drug_one = Input(
            shape=(1,), name='input_drug_one', dtype='int64')
        input_drug_two = Input(
            shape=(1,), name='input_drug_two', dtype='int64')

        drug_one_embedding = Embedding(input_dim=self.config.drug_vocab_size,
                                       output_dim=self.config.embed_dim,
                                       embeddings_initializer='glorot_normal',
                                       embeddings_regularizer=l2(
                                           self.config.l2_weight),
                                        # weights=[self.config.drug_feature],
                                       name='user_embedding')
        # assign pre-trained embedding using self.drug_feature
        # drug_one_embedding.set_weights([self.config.drug_feature])
        entity_embedding = Embedding(input_dim=self.config.entity_vocab_size,
                                     output_dim=self.config.embed_dim,
                                     embeddings_initializer='glorot_normal',
                                     embeddings_regularizer=l2(
                                         self.config.l2_weight),
                                        # weights=[self.config.drug_feature],
                                     name='entity_embedding')
        # entity_embedding.set_weights([self.config.drug_feature])
        relation_embedding = Embedding(input_dim=self.config.relation_vocab_size,
                                       output_dim=self.config.embed_dim,
                                       embeddings_initializer='glorot_normal',
                                       embeddings_regularizer=l2(
                                           self.config.l2_weight),
                                       name='relation_embedding')
                
        receptive_field_layer = ReceptiveFieldLayer(self.config)
        neighbor_info_layer = NeighborInfoLayer(self.config)
        feature_layer = FeatureLayer(self.config)
        agg = Aggregator['feature'](
            activation='tanh' ,
            regularizer=l2(2e-6),
            name=f'aggregator_feature'
        )
        score_layer = ScoreLayer(agg)
        # siamese_score_layer = SiameseScoreLayer(agg)    
        drug_embed = drug_one_embedding(
            input_drug_one)  # [batch_size, 1, embed_dim]

        # receptive_list_drug_one = Lambda(lambda x: self.get_receptive_field(x),
                                        #  name='receptive_filed_drug_one')(input_drug_one)

        # receptive_list_drug_one = self.get_receptive_field(input_drug_one)

        receptive_list_drug_one = receptive_field_layer(input_drug_one)
        
        neineigh_ent_list_drug_one = receptive_list_drug_one[:self.config.n_depth + 1]
        neigh_rel_list_drug_one = receptive_list_drug_one[self.config.n_depth + 1:]

        neigh_ent_embed_list_drug_one = [entity_embedding(
            neigh_ent) for neigh_ent in neineigh_ent_list_drug_one]
        
        # neigh_rel_embed_list_drug_one = [tf.linalg.tensor_diag(tf.linalg.diag_part(embed_diag_layer(relation_embedding(
        #     neigh_rel)))) for neigh_rel in neigh_rel_list_drug_one]

        neigh_rel_embed_list_drug_one = [relation_embedding(
            neigh_rel) for neigh_rel in neigh_rel_list_drug_one]
    
        
        # neighbor_embedding = Lambda(lambda x: self.get_neighbor_info(x[0], x[1], x[2]),
        #                             name='neighbor_embedding_drug_one')

        for depth in range(self.config.n_depth):
            aggregator = Aggregator[self.config.aggregator_type](
                activation='tanh' if depth == self.config.n_depth - 1 else 'relu',
                regularizer=l2(self.config.l2_weight),
                name=f'aggregator_{depth + 1}_drug_one'
            )

            next_neigh_ent_embed_list_drug_one = []
            for hop in range(self.config.n_depth - depth):
                # neighbor_embed = neighbor_embedding([drug_embed, neigh_rel_embed_list_drug_one[hop],
                #                                      neigh_ent_embed_list_drug_one[hop + 1]])
                # neighbor_embed = self.get_neighbor_info(drug_embed, neigh_rel_embed_list_drug_one[hop],
                #                                         neigh_ent_embed_list_drug_one[hop + 1])
                neighbor_embed = neighbor_info_layer([drug_embed, neigh_rel_embed_list_drug_one[hop],
                                                        neigh_ent_embed_list_drug_one[hop + 1]])
                next_entity_embed = aggregator(
                    [neigh_ent_embed_list_drug_one[hop], neighbor_embed])
                next_neigh_ent_embed_list_drug_one.append(next_entity_embed)
            neigh_ent_embed_list_drug_one = next_neigh_ent_embed_list_drug_one

        # drug1_feature = Lambda(lambda x:self.getfeature(x),name = 'feature1')(input_drug_one)
        # drug1_feature = self.getfeature(input_drug_one)
        drug1_feature = feature_layer(input_drug_one)
        drug1_embed = neigh_ent_embed_list_drug_one[0]

        # get receptive field
        # receptive_list = Lambda(lambda x: self.get_receptive_field(x),
        #                         name='receptive_filed')(input_drug_two)
        # receptive_list = self.get_receptive_field(input_drug_two)
        receptive_list = receptive_field_layer(input_drug_two)
        neigh_ent_list = receptive_list[:self.config.n_depth + 1]
        neigh_rel_list = receptive_list[self.config.n_depth + 1:]

        neigh_ent_embed_list = [entity_embedding(
            neigh_ent) for neigh_ent in neigh_ent_list]
        neigh_rel_embed_list = [relation_embedding(
            neigh_rel) for neigh_rel in neigh_rel_list]
        # neigh_rel_embed_list = [tf.linalg.tensor_diag(tf.linalg.diag_part(embed_diag_layer(relation_embedding(
        #     neigh_rel)))) for neigh_rel in neigh_rel_list]

        # neighbor_embedding = Lambda(lambda x: self.get_neighbor_info(x[0], x[1], x[2]),
        #                             name='neighbor_embedding')

        for depth in range(self.config.n_depth):
            aggregator = Aggregator[self.config.aggregator_type](
                activation='tanh' if depth == self.config.n_depth - 1 else 'relu',
                regularizer=l2(self.config.l2_weight),
                name=f'aggregator_{depth + 1}'
            )

            next_neigh_ent_embed_list = []
            for hop in range(self.config.n_depth - depth):
                # neighbor_embed = neighbor_embedding([drug_embed, neigh_rel_embed_list[hop],
                #                                      neigh_ent_embed_list[hop + 1]])
                # neighbor_embed = self.get_neighbor_info(drug_embed, neigh_rel_embed_list[hop],
                #                                         neigh_ent_embed_list[hop + 1])
                neighbor_embed = neighbor_info_layer([drug_embed, neigh_rel_embed_list[hop],
                                                        neigh_ent_embed_list[hop + 1]])
                next_entity_embed = aggregator(
                    [neigh_ent_embed_list[hop], neighbor_embed])
                next_neigh_ent_embed_list.append(next_entity_embed)
            neigh_ent_embed_list = next_neigh_ent_embed_list

        # drug2_feature = Lambda(lambda x:self.getfeature(x),name = 'feature2')(input_drug_two)
        # drug2_feature = self.getfeature(input_drug_two)
        
        #############################################
        #############################################
        drug2_feature = feature_layer(input_drug_two)
        drug2_embed = neigh_ent_embed_list[0]

        # drug_drug_score = Lambda(lambda x: self.getscore(x),name='score')(
        #     [drug1_embed, drug2_embed, drug1_feature, drug2_feature])
        # drug_drug_score = self.getscore(
        #     [drug1_embed, drug2_embed, drug1_feature, drug2_feature])
        drug_drug_score = score_layer([drug1_embed, drug2_embed, drug1_feature, drug2_feature])
        # drug_drug_score = siamese_score_layer([drug1_embed, drug2_embed, drug1_feature, drug2_feature])
        # print(drug_drug_score)

        model = Model([input_drug_one, input_drug_two], drug_drug_score)
        model.compile(optimizer=self.config.optimizer,
                      loss='binary_crossentropy', metrics=['acc'])

        return model

    def get_receptive_field(self, entity):
        """Calculate receptive field for entity using adjacent matrix

        :param entity: a tensor shaped [batch_size, 1]
        :return: a list of tensor: [[batch_size, 1], [batch_size, neighbor_sample_size],
                                   [batch_size, neighbor_sample_size**2], ...]
        """
        neigh_ent_list = [entity] # 1 
        neigh_rel_list = []
        adj_entity_matrix = K.variable(
            self.config.adj_entity, name='adj_entity', dtype='int64')
        adj_relation_matrix = K.variable(self.config.adj_relation, name='adj_relation',
                                         dtype='int64')
        n_neighbor = K.shape(adj_entity_matrix)[1]

        for i in range(self.config.n_depth):
            new_neigh_ent = K.gather(adj_entity_matrix, K.cast(
                neigh_ent_list[-1], dtype='int64'))  # cast function used to transform data type
            new_neigh_rel = K.gather(adj_relation_matrix, K.cast(
                neigh_ent_list[-1], dtype='int64'))
            neigh_ent_list.append(
                K.reshape(new_neigh_ent, (-1, n_neighbor ** (i + 1))))
            neigh_rel_list.append(
                K.reshape(new_neigh_rel, (-1, n_neighbor ** (i + 1))))

        return neigh_ent_list + neigh_rel_list

    def get_neighbor_info(self, drug, rel, ent):
        """Get neighbor representation.

        :param user: a tensor shaped [batch_size, 1, embed_dim]
        :param rel: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        :param ent: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        :return: a tensor shaped [batch_size, neighbor_size ** (hop -1), embed_dim]
        """
        # [batch_size, neighbor_size ** hop, 1] drug-entity score
        drug_rel_score = K.sum(drug * rel, axis=-1, keepdims=True)

        # [batch_size, neighbor_size ** hop, embed_dim]
        weighted_ent = drug_rel_score * ent

        # [batch_size, neighbor_size ** (hop-1), neighbor_size, embed_dim]
        weighted_ent = K.reshape(weighted_ent,
                                 (K.shape(weighted_ent)[0], -1,
                                  self.config.neighbor_sample_size, self.config.embed_dim))

        neighbor_embed = K.sum(weighted_ent, axis=2)
        return neighbor_embed

    def getfeaturetensor(self):
        drug = self.config.drug_feature
        return K.variable(drug,dtype='float32')

    def getfeature(self,input_drug):
        #read drug feature vector
        #K.gather return[batch_size,drug_feature_embed_dimension]
        drug = [input_drug][-1]
        drug_feature_tensor = K.variable(self.config.drug_feature,dtype='float32')
        drug_f = K.gather(drug_feature_tensor, K.cast(drug, dtype='int64'))
        return drug_f

    def getscore(self, embed_list):
        drug1 = embed_list[0]  # [batch_size,,embed_size]
        drug2 = embed_list[1]
        drug1_f = embed_list[2]  # [batch_size,embed_size]
        drug2_f = embed_list[3]


        aggregator = Aggregator['feature'](
            activation='tanh' ,
            regularizer=l2(2e-6),
            name=f'aggregator_feature'
        )

        # drug1 = aggregator([drug1,drug1_f])
        # drug2 = aggregator([drug2,drug2_f])

        drug1 = K.squeeze(drug1,axis=1)
        drug2 = K.squeeze(drug2,axis=1)

        drug_drug_score = K.sigmoid(K.sum(drug1 * drug2, axis=-1, keepdims=True))


        return drug_drug_score

    def add_metrics(self, x_train, y_train, x_valid, y_valid, out_folder):
        self.callbacks.append(KGCNMetric(x_train, y_train, x_valid, y_valid,
                                         self.config.aggregator_type, self.config.dataset, self.config.K_Fold,
                                         self.config.batch_size, out_folder))
    
    def add_tboard_callback(self):
        self.callbacks.append(self.config.callbacks_tboard)
        print('Logging Info - Callback Added: KGCNMetric...')
    
    def add_lr_decay_callback(self):
        self.callbacks.append(self.config.callbacks_lr_decay)
        print('Logging Info - Callback Added: LR Decay...') 
        
    def fit(self, x_train, y_train, x_valid, y_valid, out_folder):
        self.callbacks = []
        self.add_metrics(x_train, y_train, x_valid, y_valid, out_folder)
        self.add_tboard_callback()
        # self.add_lr_decay_callback()
        self.init_callbacks(out_folder)
        print('Logging Info - Start training...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size,
                       epochs=self.config.n_epoch, validation_data=(x_valid, y_valid),
                       callbacks=self.callbacks)
        print('Logging Info - training end...')

    def predict(self, x):
        return self.model.predict(x, batch_size=self.config.batch_size).flatten()

    def score(self, x, y, threshold=0.5):

        y_true = y.flatten()
        y_pred = self.model.predict(x, batch_size=self.config.batch_size).flatten()
        # auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        try:
            auc = roc_auc_score(y_true=y_true, y_score=y_pred)  # roc曲线的auc
        except ValueError:
            auc = 1
            pass
        p, r, t = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
        aupr = m.auc(r, p)
        y_pred_2 = [1 if prob >= threshold else 0 for prob in y_pred]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred_2)
        f1 = f1_score(y_true=y_true, y_pred=y_pred_2)

        return y_pred,y_pred_2,auc, acc, f1, aupr
