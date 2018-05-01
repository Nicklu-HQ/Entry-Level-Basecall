#!/usr/bin/python

import tensorflow   as tf
import numpy        as np

from   get_data_01  import get_batch_data
from   get_data_01  import CTC_sparse, decode_sparse

fold_h5  = r'D:\ONT_simulator\lamda_0812_pass_albacore\workspace\pass\0'
# change this fold to the your data fold path


num_events = 200
time_steps = num_events*7
batch_size = 10


SaveModel  =  './Saver/e2e'
Lr         = 0.05


#------------------------------- basecaller------------------------------
def Classifor(in_data,training=True, reuse=False):
    #  input_tensor.shape  = [batch_size,time_steps,1]
    #  output_tensor.shape = [batch_size,time_steps,5]    
    with tf.variable_scope('classifor',reuse=reuse):
        Layer1_1    = tf.layers.conv1d(in_data,  filters=50, kernel_size=50, strides=1, padding="same",activation=None,name='C_L1_cat_1')
        BN_Layer1_1 = tf.contrib.layers.batch_norm(Layer1_1,activation_fn=tf.nn.relu,is_training=training)

        Layer1_2 = tf.layers.conv1d(BN_Layer1_1, filters=100, kernel_size=50, strides=1, padding="same",activation=None,name='C_L1_cat_2')
        BN_Layer1_2 = tf.contrib.layers.batch_norm(Layer1_2,activation_fn=tf.nn.relu,is_training=training)
        
        Layer1_3 = tf.layers.conv1d(BN_Layer1_2, filters=100, kernel_size=50, strides=1, padding="same",activation=None,name='C_L1_cat_3')
        BN_Layer1_3 = tf.contrib.layers.batch_norm(Layer1_3,activation_fn=tf.nn.relu,is_training=training)

        BN_Layer2 = tf.concat([in_data,BN_Layer1_1,BN_Layer1_2,BN_Layer1_3,],2)

        Layer3 = tf.layers.conv1d(BN_Layer2, filters=150, kernel_size=50, strides=1, padding="same",activation=None,name='C_L_3')
        BN_Layer3 = tf.contrib.layers.batch_norm(Layer3, activation_fn=tf.nn.relu, is_training=training)
        
        Layer4 = tf.layers.conv1d(BN_Layer3, filters=64,  kernel_size=20, strides=1, padding="same",activation=None,name='C_L_4')
        BN_Layer4 = tf.contrib.layers.batch_norm(Layer4, activation_fn=tf.nn.relu, is_training=training)
        
        Layer5 = tf.layers.conv1d(BN_Layer4, filters=5,   kernel_size=1, strides=1, padding="same",activation=None,name='C_L_5')        
        #Layer5.shape = [batch_size,time_steps,5]

        pred_class = tf.reshape(Layer5,[-1,5]) 
        
        out    = tf.transpose(Layer5,(1,0,2))
        #out.shape    = [time_steps,batch_size,5] 
        return out,pred_class

#------------------------------- network ------------------------------
with tf.name_scope('inputs'):
    x    = tf.placeholder(tf.float32,shape=[None,time_steps,1], name="inputX")    
    y    = tf.sparse_placeholder(tf.int32, name="inputY")
    seq_length=tf.placeholder(tf.int32,[None])

pred,pred_class = Classifor(x,training=True, reuse=False)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.name_scope('loss_'):
    loss=tf.reduce_mean(tf.nn.ctc_loss(labels=y,inputs=pred,sequence_length=seq_length))
    
with tf.name_scope('train'):
    with tf.control_dependencies(update_ops):   
        training      = tf.train.AdamOptimizer(learning_rate=Lr).minimize(loss)

with tf.name_scope('distance'):
    decoded,log_prob=tf.nn.ctc_beam_search_decoder(pred,seq_length,merge_repeated=False)
    distance=tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0],tf.int32),y))


label_true = tf.sparse_tensor_to_dense(y,         default_value=5)
label_pred = tf.sparse_tensor_to_dense(decoded[0],default_value=5)

#------------------------------- training ------------------------------

print ("Training ------------")

training_step  = 200000
saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for steps in range (training_step):
        seq ,signal = get_batch_data(fold_h5,num_events,batch_size)
        seq         = np.array(seq).reshape(batch_size,-1)
        signal      = signal.reshape(batch_size,time_steps,1)
        
        batch_indices,batch_values,batch_shapes = CTC_sparse(seq,batch_size)
        feed = {x:signal, y:(batch_indices,batch_values,batch_shapes),seq_length:np.ones(batch_size)*time_steps}
        sess.run(training,feed)

        if steps%50 == 0:
            Dis_v,loss_v,y_ture,y_pred = sess.run([distance,loss,label_true,label_pred],feed)

            print ("After %d step ------- Distance and Cost are: "%(steps), Dis_v,   "  and   " ,loss_v,)    
            print ("____________________________________________________")
            print("y_ture is:   " , decode_sparse(y_ture,batch_size)[0])
            print("y_pred is:   " , decode_sparse(y_pred,batch_size)[0])
        
        # 保存
        if steps % 100 ==0:
            saver.save(sess, SaveModel)













    
