""" utils modules for transformer model """

import tensorflow as tf 
import numpy as np 

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, hidden_dim, input_vocab_size, dropout_rate=0.1):
        super(Encoder, self).__init__()

        # ** word embedding
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, self.embedding_dim)

        # ** pos encoding
        self.pos_encoding = positional_encoding(input_vocab_size, self.embedding_dim)

        # ** building N=num_layers `EncoderLayer`s
        self.encoding_layers = [EncoderLayer(self.embedding_dim, num_heads, hidden_dim, dropout_rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, is_training, mask):
        """
        x.shape == (batch_size, input_seq_length)
        """
        input_seq_length = tf.shape(x)[-1]

        x = self.embedding(x) # NOTE x.shape == (batch_size, seq_length, embedding_dim)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        x += self.pos_encoding[:, :input_seq_length, :]

        x = self.dropout(x, training=is_training)

        for _, enc_layer in enumerate(self.encoding_layers):
            x = enc_layer(x, is_training, mask)

        return x

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout_rate):
        super(EncoderLayer, self).__init__()
        
        self.hma = MutiHeadAttention(embedding_dim, num_heads)
        self.ffn = feed_forward_network(embedding_dim, hidden_dim)

        # ** layer normalize
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # ** dropout layers
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, is_training, mask):
        """
        attn.shape == (batch_size, num_heads, input_seq_length, input_seq_length)
        """
        # ** MHA
        atten_output, atten = self.hma(x, x, x, mask) # NOTE self attention >> q = k = v
        atten_output = self.dropout1(atten_output, training=is_training)
        out1 = self.layernorm1(x + atten_output)
        
        # ** FFN
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=is_training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class MutiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads):
        super(MutiHeadAttention, self).__init__()
        print('embedding_dim = ', embedding_dim)
        print('num_heads = ', num_heads)
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim # NOTE dimension before split heads
        
        self.depth = embedding_dim // num_heads

        self.wq = tf.keras.layers.Dense(embedding_dim)   
        self.wk = tf.keras.layers.Dense(embedding_dim)  
        self.wv = tf.keras.layers.Dense(embedding_dim)

        self.dense = tf.keras.layers.Dense(embedding_dim)  # NOTE after heads concat

    def _split_heads(self, x, batch_size):
        """
        (1) Split the last dimension into (num_heads, depth)
        (2) transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # ** to embedding
        q = self.wq(q) # NOTE to (batch_size, seq_length_q, embedding_dim)
        k = self.wk(k)
        v = self.wv(v)

        # ** split heads NOTE embedding_dim to (num_heads, depth)
        q = self._split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self._split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self._split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weigths = get_attention(q, k, v, mask)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        # ** to (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # ** concate attentions
        concated_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embedding_dim))

        # ** output layer
        outout = self.dense(concated_attention)

        return outout, attention_weigths

def feed_forward_network(output_dim, hidden_dim):
    return tf.keras.Sequential([tf.keras.layers.Dense(hidden_dim, activation='relu'),
                                tf.keras.layers.Dense(output_dim)])

def get_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True) # NOTE (..., seq_length_q, seq_length_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32) # NOTE length of seq k
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # NOTE k axis
    
    # ** weighted v
    output = tf.matmul(attention_weights, v) # NOTE (..., seq_length_q, depth_v)

    return output, attention_weights

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout_rate=0.0):
        super(DecoderLayer, self).__init__()

        # ** sub-layers
        self.mha1 = MutiHeadAttention(embedding_dim, num_heads) # NOTE self-attention
        self.mha2 = MutiHeadAttention(embedding_dim, num_heads) # NOTE encode-decode attention
        self.ffn = feed_forward_network(embedding_dim, hidden_dim)

        # ** LayerNorm
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # ** dropout layers
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, is_training, combined_mask, inp_padding_mask):

        # ** self-attention
        atten1, atten_weights1 = self.mha1(x, x, x, combined_mask)
        atten1 = self.dropout1(atten1, training=is_training)
        out1 = self.layernorm1(atten1 + x)

        # ** attention to enc_outputs
        atten2, atten_weights2 = self.mha2(enc_output, enc_output, out1, inp_padding_mask)  # NOTE (batch_size, target_seq_len, embedding_dim)
        atten2 = self.dropout2(atten2, training=is_training)                                # v, k, q
        out2 = self.layernorm2(atten2 + out1)  # (batch_size, target_seq_len, embedding_dim)

        # ** FFN
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=is_training)
        
        out3 = self.layernorm3(ffn_output + out2)

        return out3, atten_weights1, atten_weights2

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, hidden_dim, target_vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, embedding_dim)
        self.pos_encoding = positional_encoding(target_vocab_size, embedding_dim)
        self.decode_layers = [DecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, is_training, combined_mask, inp_padding_mask):
        target_seq_len = tf.shape(x)[-1]
        attention_weights_dict = {}

        x = self.embedding(x) # NOTE x.shape == (batch_size, seq_length, embedding_dim)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        x += self.pos_encoding[:, :target_seq_len, :]
        x = self.dropout(x, training=is_training)

        for i, dec_layer in enumerate(self.decode_layers):
            x, w1, w2 = dec_layer(x, enc_output, is_training, combined_mask, inp_padding_mask)
            attention_weights_dict['decoder_layer_{}_w1'.format(i + 1)] = w1
            attention_weights_dict['decoder_layer_{}_w1'.format(i + 1)] = w2

        return x, attention_weights_dict

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  sines = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  cosines = np.cos(angle_rads[:, 1::2])
  
  pos_encoding = np.concatenate([sines, cosines], axis=-1)
  
  pos_encoding = pos_encoding[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

class Transformer(tf.keras.models.Model):
    def __init__(self, num_layers, embedding_dim, num_heads, hidden_dim, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, embedding_dim, num_heads, hidden_dim, input_vocab_size, dropout_rate)
        self.decoder = Decoder(num_layers, embedding_dim, num_heads, hidden_dim, target_vocab_size, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, target, is_training, enc_padding_mask, combined_mask, dec_padding_mask):
        enc_output = self.encoder(inputs, is_training, enc_padding_mask) # NOTE (batch_size, tar_seq_len, target_vocab_size)
        dec_output, attention_weights = self.decoder(target, enc_output, is_training, combined_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)  # NOTE (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights


