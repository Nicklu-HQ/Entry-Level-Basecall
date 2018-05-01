#!/usr/bin/python

import h5py
import numpy as np
import os
import random


######################### help Function ############################

def CTC_sparse(label,batch_size):
    dict_ = {'A':0,'G':1,'C':2,'T':3}
    batch_shapes     = []
    batch_indices    = []
    batch_values     = []
    batch_count      = 0
    for j in range(batch_size):
        for i,item in enumerate(label[j][0]):
            batch_indices.append((batch_count,i))
            batch_values.append(dict_[item])
        batch_count+=1
    batch_shapes.append(np.asarray([batch_size,np.asarray(batch_indices,np.int64).max(0)[1]+1],dtype=np.int64))
    batch_indices = np.array(batch_indices)
    batch_values  = np.array(batch_values)
    batch_shapes  = np.array(batch_shapes[0])
       
    return batch_indices,batch_values,batch_shapes


def decode_sparse(y,batch_size):
    #用了tf.sparse_tensor_to_dense(targets,default_value=62)
    characters  =['1','2','3','4','n','m','z']
    label_total = []
    for i in range(batch_size):
        label_tem = ''.join([characters[x] for x in y[i]])
        label_total.append(label_tem)
    return label_total




######################## inputdata Function ###################


def get_random_one_file(fold_path):     
    file_all = os.listdir(fold_path)   
    max_file = len(file_all)
    file_name = file_all[random.randint(0, max_file - 1)]
    file_path = os.path.join(fold_path,file_name)
    return file_name,file_path

def printname(name):
    print(name)

def get_data_from_file(fold_path):
    file_name,file_path = get_random_one_file(fold_path)
    if file_path == None:
        return None,None
    with h5py.File(file_path) as h5:
        #h5.visit(printname)
        read_num     = file_name.split('_')[10]
        path_signal  = 'Raw/Reads/Read_'+ read_num +'/Signal'
        path_events  = 'Analyses/Basecall_1D_001/BaseCalled_template/Events'
        signal  = h5[path_signal].value
        events  = h5[path_events].value
    return signal, events

def get_part_data(fold_path,num_events):
    signal, events = get_data_from_file(fold_path)
    if len(events) <= (num_events + 1000):
        return None, None
    T     = len(events)
    left  = 100                                       #每个events对应7个points
    start = random.randint(left, T-left-num_events)   #设定一个截取的范围
    end   = start + num_events
    events_part = events[start:end]
    seq         = ''  
    for event in events_part:
        number  = int(event[5])
        if number > 0:
            seq_t   = str(event[4], encoding='utf-8')
            seq_n   = seq_t[-number:]
            seq    += seq_n        
    start_signal = int(events_part[0][1])
    signal_part  = signal[start_signal:start_signal+num_events*7]
    return seq ,signal_part


def get_batch_data(fold_path,num_events,batch_size):
    seq_all     = []
    signal_all  = []
    number_temp = 0 
    while number_temp < batch_size:
        seq_match,signal_part = get_part_data(fold_path,num_events)
        if seq_match != None and len(seq_match) > 100:    # 避免匹配不上的部分
            seq_all.append(seq_match)
            signal_all.append(signal_part)
            number_temp += 1
    seq_all    = np.array(seq_all)
    signal_all = np.array(signal_all)
    return seq_all ,signal_all
    
#___________________________________________
"""
fold_h5    = r'D:\ONT_simulator\lamda_0812_pass_albacore\workspace\pass\0'
num_events = 100
batch_size = 5

seq ,signal_part =get_batch_data(fold_h5,num_events,batch_size)



print(seq[2])
print(signal_part.shape)
"""







