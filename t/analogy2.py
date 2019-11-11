from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
from scipy import misc as spm
import pickle
from pathlib import Path

class NeuralAnalogy:
    
    def __construct_recode_mat_aux(self, wm_i, state):
            
        if (wm_i >= self._wm_c):
            yield(state)
            return
        
        for t in [x for x in range(self._wm_c) if not (x in state)]:
            state_t1 = np.copy(state)
            state_t1[wm_i] = t
       
            for state_t in self.__construct_recode_mat_aux(wm_i + 1, state_t1):
                yield state_t
    
        
    def __construct_recode_mat(self):
        diag_mat = np.diagflat(np.ones(shape=(self._wm_c)))
        recode_mat = np.zeros(shape=(self._n_states, self._wm_c, self._wm_c), dtype=np.float32)
        state = np.full(shape=self._wm_c, fill_value=-1, dtype=np.int)
        i = 0
        for x in self.__construct_recode_mat_aux(0, state):
            recode_mat[i] = diag_mat[:][x]
            i += 1
        return recode_mat
    
    
    def __create_graph(self):
        
        self.__recode_mat = tf.placeholder(shape=[self._n_states, self._wm_c, self._wm_c], dtype=tf.float32)
        self.__sem_target = tf.placeholder(shape=[self._wm_c, self._sem_dim], dtype=tf.float32)
        self.__struct_target = tf.placeholder(shape=[self._max_ar, self._wm_c, self._wm_c], dtype=tf.float32)
 
        self.__sem_base = tf.placeholder(shape=[None, self._wm_c, self._sem_dim], dtype=tf.float32)
        self.__struct_base = tf.placeholder(shape=[None, self._max_ar, self._wm_c, self._wm_c], dtype=tf.float32)        
            
        #generate all possible states of the semantics of the target
        sem_outputs = tf.reshape(                
                tf.matmul(
                        tf.reshape(
                                self.__recode_mat, 
                                [self._n_states * self._wm_c,self._wm_c]), 
                        self.__sem_target),
                [1, self._n_states, self._wm_c * self._sem_dim]
            )
        
        #generate all possible states of the structure of the target
        struct_outputs = tf.reshape( 
            tf.transpose(
                tf.reshape(
                    tf.concat(
                        [                            
                            tf.matmul(
                                tf.reshape(
                                    tf.matmul(
                                        tf.reshape(self.__recode_mat, [self._n_states * self._wm_c, self._wm_c]), 
                                        self.__struct_target[a_i],
                                    ), 
                                    [self._n_states, self._wm_c, self._wm_c]
                                ),
                                tf.transpose(self.__recode_mat, [0, 2, 1]),
                            )
                            for a_i in range(self._max_ar)
                        ], 0
                    ), [self._max_ar, self._n_states, self._wm_c * self._wm_c]
                 ), 
                [1, 0, 2])
            ,
            [1, self._n_states, self._max_ar * self._wm_c * self._wm_c])
        
        n_bases = tf.shape(self.__sem_base)[0]
        
        sem_base = tf.reshape(
            tf.tile(
                    tf.reshape(
                            self.__sem_base, 
                            [n_bases , self._wm_c * self._sem_dim]),
                    [1, self._n_states]),
            [n_bases, self._n_states, self._wm_c * self._sem_dim])
        
        struct_base = tf.reshape(
            tf.tile(
                    tf.reshape(
                            self.__struct_base, 
                            [n_bases, self._max_ar * self._wm_c * self._wm_c]),
                    [1, self._n_states]),
            [n_bases, self._n_states, self._max_ar * self._wm_c * self._wm_c])
        
        
        denom_sem = tf.multiply( 
            tf.sqrt(tf.reduce_sum(tf.multiply(sem_outputs, sem_outputs), axis=[2])),                
            tf.sqrt(tf.reduce_sum(tf.multiply(sem_base, sem_base), axis=[2]))
        )      
        num_sem = tf.reduce_sum(tf.multiply(sem_outputs, sem_base), axis=[2])        
                
        sem_cos = tf.add(tf.multiply(tf.divide(num_sem, denom_sem), 0.5), 0.5) 
                
#        denom_struct = tf.multiply( 
#            tf.sqrt(tf.reduce_sum(tf.multiply(struct_outputs, struct_outputs), axis=[2])),                
#            tf.sqrt(tf.reduce_sum(tf.multiply(struct_base, struct_base), axis=[2]))
#        )        
        denom_struct = tf.reduce_sum(struct_outputs, axis=[2])
        
        num_struct = tf.reduce_sum(tf.multiply(struct_outputs, struct_base), axis=[2])        
        struct_cos = tf.divide(num_struct, denom_struct)
        #struct_cos = tf.divide(num_struct, )
        
        similarities = tf.add(
                tf.multiply(sem_cos, 1 - self._sigma), 
                tf.multiply(struct_cos, self._sigma)
            )

        self._sem_cos = sem_cos
        self._struct_cos = struct_cos
        
        base_max_similarities = tf.reduce_max(similarities, axis=[1])
        self.__best_base_index = tf.argmax(base_max_similarities)
        best_recoding_no = \
            tf.argmax(
                tf.reshape(
                        tf.slice(
                                similarities,
                                [self.__best_base_index, 0],[1, self._n_states]),
                        [self._n_states]))

        self.__best_base_similarity = tf.reduce_max(base_max_similarities)                
        
        self.__best_recoding = tf.slice(
                self.__recode_mat, 
                [best_recoding_no, 0, 0], 
                [1, self._wm_c, self._wm_c])
        
        self.__best_target_sem_recoding = tf.reshape(
            tf.slice(sem_outputs, [0, best_recoding_no, 0], [1, 1, self._wm_c * self._sem_dim]),
            [self._wm_c, self._sem_dim])
        
        self.__best_target_struct_recoding = tf.reshape(
            tf.slice(struct_outputs, [0, best_recoding_no, 0], [1, 1, self._max_ar * self._wm_c * self._wm_c]),
            [self._max_ar, self._wm_c, self._wm_c])
        
    
    def make(self, target, bases):
        """
            Returns the predicate type id which corresponds to the given semantic vector.
            Use to recover the predicate type from a vector representation
        """      
        if not (type(bases) is list):
            sem_bases = [bases[0]]
            struct_bases = [bases[1]]
        else:
            sem_bases = np.zeros(shape=(len(bases), self._wm_c, self._sem_dim), dtype=np.float32)
            struct_bases = np.zeros(shape=(len(bases), self._max_ar, self._wm_c, self._wm_c), dtype=np.float32)
            for i in range(len(bases)):
                sem_bases[i] = bases[i][0]
                struct_bases[i] = bases[i][1]
                
                
        #Construct recoding matrix
        if Path("recode/recode_mat.{}.pickle".format(self._wm_c)).is_file():
            #a recoding matrix with given parameters is already created and serialized, load it
            print("Loading recoding matix...", end="")
            sys.stdout.flush()
            with open("recode/recode_mat.{}.pickle".format(self._wm_c), "r") as f:            
                recode_mat = pickle.load(f)
            print("Done.")
            sys.stdout.flush()
        else:
            #create recoding matrix and serialize it to a file
            recode_mat = self.__construct_recode_mat()
            with open("recode/recode_mat.{}.pickle".format(self._wm_c), "w") as f:
                with tf.Session():
                    pickle.dump(recode_mat.eval(), f)                    
        states_processed = 0
        with tf.Session() as session:
            max_sim = 0
            best_base_index = 0
            best_recoding = 0
            while states_processed < int(spm.factorial(self._wm_c)):            
                result = \
                    session.run(
                            [
                                self.__best_base_similarity, 
                                self.__best_base_index,
                                self.__best_recoding,
                                self._sem_cos,
                                self._struct_cos
                            ],
                            feed_dict={
                                    self.__sem_target: target[0],
                                    self.__struct_target: target[1],
                                    self.__sem_base: sem_bases,
                                    self.__struct_base: struct_bases,
                                    self.__recode_mat: recode_mat[states_processed:states_processed+self._n_states,:,:]
                            })
    
                states_processed += self._n_states
                if (result[0] > max_sim):
                    max_sim = result[0]
                    best_base_index = result[1]
                    best_recoding = result[2]
            
        return (max_sim, best_base_index, best_recoding)
        

            
    def __init__(self, sigma, wm_c, max_ar, sem_dim, n_states = 0):
        self._sigma = sigma
        self._wm_c = wm_c
        self._max_ar = max_ar
        self._sem_dim = sem_dim
        #number of possible states of the representations
        if (n_states == 0):
            self._n_states = int(spm.factorial(wm_c))    
        else:
            self._n_states = n_states
        self.__create_graph()
            