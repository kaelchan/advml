import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Activation
from keras.models import Model


def dense_batch_fc_tanh(x, units, phase, scope, do_norm=False):
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.zeros_initializer()
    h1 = Dense(units, kernel_initializer = w_init, bias_initializer = b_init)(x)
    if do_norm:
        h2 = BatchNormalization(momentum = 0.9, center=True, scale=True, training=phase)(h1)
        return Activation('tanh')(h2)
    else:
        return Activation('tanh')(h1)

class DeepCF:
    """
    main model class implementing DeepCF
    also stores states for fast candidate generation

    latent_rank_in: rank of preference model input
    user_content_rank: rank of user content input
    item_content_rank: rank of item content input
    model_select: array of number of hidden unit,
        i.e. [200,100] indicate two hidden layer with 200 units followed by 100 units
    rank_out: rank of latent model output

    """

    def __init__(self, latent_rank_in, user_content_rank, item_content_rank, model_select, rank_out):

        self.rank_in = latent_rank_in
        self.phi_u_dim = user_content_rank
        self.phi_v_dim = item_content_rank
        self.model_select = model_select
        self.rank_out = rank_out

        # inputs
        self.Uin = None
        self.Vin = None
        self.Ucontent = None
        self.Vcontent = None
        self.phase = None
        self.target = None
        self.eval_trainR = None
        self.U_pref_tf = None
        self.V_pref_tf = None
        self.rand_target_ui = None

        # outputs in the model

        self.preds = None
        #self.updates = None
        self.loss = None
        self.model = None

        self.U_embedding = None
        self.V_embedding = None

        self.lr_placeholder = None

        # predictor
        self.tf_topk_vals = None
        self.tf_topk_inds = None
        self.preds_random = None
        self.tf_latent_topk_cold = None
        self.tf_latent_topk_warm = None
        self.eval_preds_warm = None
        self.eval_preds_cold = None

    def build_model(self):
        self.Vin = Input(shape=(self.rank_in,), dtype='float32', name='V_in_raw')
        self.Uin = Input(shape=(self.rank_in,), dtype='float32', name='U_in_raw')
        self.Vcontent = Input(shape=(self.phi_v_dim,), dtype='float32', name='V_content')
        self.Ucontent = Input(shape=(self.phi_u_dim,), dtype='float32', name='U_content')
        u_concat = tf.concat([self.Uin, self.Ucontent], 1)
        v_concat = tf.concat([self.Vin, self.Vcontent], 1)

        for ihid, hid in enumerate(self.model_select):
            u_last = dense_batch_fc_tanh(u_concat, hid, self.phase, 'user_layer_%d' % (ihid + 1), do_norm=True)
            v_last = dense_batch_fc_tanh(v_concat, hid, self.phase, 'item_layer_%d' % (ihid + 1), do_norm=True)

        u_emb_w_init = tf.truncated_normal([u_last.get_shape().as_list()[1], self.rank_out], stddev=0.01)
        u_emb_b_init = tf.zeros([1, self.rank_out])
        self.U_embedding = Dense(self.rank_out, kernel_initializer = u_emb_w_init, bias_initializer = u_emb_b_init)
        v_emb_w_init = tf.truncated_normal([u_last.get_shape().as_list()[1], self.rank_out], stddev=0.01)
        v_emb_b_init = tf.zeros([1, self.rank_out])
        self.V_embedding = Dense(self.rank_out, kernel_initializer = v_emb_w_init, bias_initializer = v_emb_b_init)
        preds = tf.multiply(self.U_embedding, self.V_embedding)
        self.preds = tf.reduce_sum(preds, 1)
        model = Model(inputs=[self.Uin, self.Vin, self, Ucontent, self.Vcontent], outputs=self.preds)
        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        self.model = model
 
 
