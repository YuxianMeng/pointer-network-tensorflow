# -*- coding: utf-8 -*-
"""
Pointer Networks
@author: Yuxian Meng
http://arxiv.org/abs/1506.03134

"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib import layers
from six.moves import xrange


def ptr_net(x, lengths, hidden_size, keep_prob, start_symbol):
    """pointer networks
    Args:
        x: tensor of shape [batch_size, max_length, input_size]
        lengths: real length of x, tensor of shape [batch_size, ]
        hidden_size: hidden_size of RNN cell
        
    Returns:
        tensor of shape [batch_size, input_size, max_length]
        
    """
    batch_size, max_len, input_size = [x.shape[i].value for i in range(3)]
    init = tf.random_normal_initializer(0.0, 0.5)
    

    # bi-RNN encoder
    with tf.variable_scope("encoder") as scope:
        cell_enc_fw = tf.contrib.rnn.LSTMCell(hidden_size,initializer = init,)   
        cell_enc_bw = tf.contrib.rnn.LSTMCell(hidden_size,initializer = init,)
        fw_drop = tf.contrib.rnn.DropoutWrapper(cell_enc_fw,
                                                input_keep_prob = keep_prob,
                                                output_keep_prob = 1.0)
        bw_drop = tf.contrib.rnn.DropoutWrapper(cell_enc_bw,
                                                input_keep_prob = keep_prob,
                                                output_keep_prob = 1.0)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                                                             cell_fw = fw_drop,
                                                             cell_bw = bw_drop, 
                                                             inputs = x,
                                                             dtype=tf.float32,
                                                             sequence_length = lengths,
                                                             scope = scope)
        enc_outputs = tf.concat(outputs, 2) 
        output_state = tf.concat(output_states, 2)
    
#    return outputs, output_states
        
    # decoder
    with tf.variable_scope("decoder") as scope:
        cell_dec = tf.contrib.rnn.LSTMCell(hidden_size*2, initializer = init,)   
        dec_state = tf.contrib.rnn.LSTMStateTuple(c = output_state[0],
                                                  h = output_state[1])
        enc_states = enc_outputs
        ptr_outputs = []
        ptr_output_dists = []
        with tf.variable_scope("rnn_decoder"):
            input_ = start_symbol

            # Push out each index
            for i in xrange(max_len):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                # Map the raw input to the LSTM dimensions
                W_d_in = tf.get_variable("W_d_in", [input_size, hidden_size*2], initializer=init)   
                b_d_in = tf.get_variable("b_d_in", [batch_size, hidden_size*2], initializer=init)
                cell_input = tf.nn.elu(tf.matmul(input_, W_d_in) + b_d_in)                            
                output, dec_state = cell_dec(cell_input, dec_state)         # Output: B x L    Dec State.c = B x L

                # Enc/dec states (.c) are B x S
                # We want to map these to 1, right?  BxS and something that maps to B alone
                num_blend_units = hidden_size #TODO: revise it
                W_1 = tf.get_variable("W_1", [hidden_size*2, num_blend_units], initializer=init)          
                W_2 = tf.get_variable("W_2", [hidden_size*2, num_blend_units], initializer=init)         
                bias_ptr = tf.get_variable("bias_ptr", [batch_size, num_blend_units], initializer=init) 

                index_predists = []
                dec_portion = tf.matmul(dec_state[0], W_2) #[0] means cell.c, [1] means .h  

                # Vector to blend
                v_blend = tf.get_variable("v_blend", [num_blend_units, 1], initializer=init)  

                for input_length_index in xrange(max_len):
                    # Use the cell values (.c), not the output (.h) values of each state
                    # Each is B x 1, and there are J of them. Flatten to J x B
                    enc_portion = tf.matmul(enc_states[:, input_length_index, :], W_1)    
                    raw_blend = tf.nn.elu(enc_portion + dec_portion + bias_ptr)  
                    scaled_blend = tf.matmul(raw_blend, v_blend)        
                    index_predist = tf.reshape(scaled_blend, (batch_size, 1))
                    index_predists.append(index_predist)
                idx_predistribution = tf.concat(index_predists, axis = -1) # batch_size * max_len
                idx_distribution = tf.nn.softmax(idx_predistribution, dim=-1) # batch_size * max_len   
                ptr_output_dists.append(tf.reshape(idx_distribution, (batch_size, max_len, 1)))
                idx = tf.argmax(idx_distribution, 1)  # batch_size  
                # Pull out the input from that index
                ptr_output_raw = tf.concat([x[batch_index, tf.to_int32(idx[batch_index]), :] for batch_index in 
                                            range(batch_size)], axis=0)          
                ptr_output = tf.reshape(ptr_output_raw, (batch_size, input_size))       
                ptr_outputs.append(tf.reshape(ptr_output, (batch_size, 1, input_size)))
                input_ = ptr_output    # The output goes straight back in as next input
        ptr_outputs = tf.concat(ptr_outputs, axis = 1)
        idx_distributions = tf.concat(ptr_output_dists, axis = -1)

    return idx_distributions, ptr_outputs

def batch_input(batch_size, input_size, max_length):
    return np.ones((batch_size, max_length, input_size)), np.ones((batch_size,)) * (max_length)
    

if __name__ == "__main__":
    tf.reset_default_graph()
    batch_size, input_size, max_length = 16, 1, 50
    max_steps=1; hidden_size = 32; keep_rate = 0.8
    start = tf.constant(0, shape=(batch_size, input_size), dtype=tf.float32)
    
    ptr_in = tf.placeholder(tf.float32, name="ptr-in", shape=(batch_size, 
                                                              max_length,
                                                              input_size))
    input_lengths = tf.placeholder(tf.int32, name = "seq_lens", 
                                   shape = [batch_size])
    out1, out2 = ptr_net(ptr_in, input_lengths, hidden_size, keep_rate, start)

    with tf.device("/cpu:0"):
        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto())
        sess.run(init)    
        for step in xrange(max_steps):
            input_seqs, seqs_lengths = batch_input(batch_size, input_size, max_length)
            out1, out2 = sess.run([out1, out2], feed_dict = {ptr_in: input_seqs,
                                                input_lengths: seqs_lengths})
    
    
    
        


