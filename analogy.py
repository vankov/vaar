"""
Vector implementation of analogy-making
"""
from __future__ import print_function
import sys
import pickle
from pathlib import Path

import tensorflow.compat.v1 as tf
import numpy as np
from scipy.special import factorial

class VectorAnalogy:
    """
    Vector analogy
    """
    def __construct_recode_mat_aux(self, slot_no, state):
        """
            Auxiliary function
        """
        if slot_no >= self._n_slots:
            yield state

        for slot_i in [x for x in range(self._n_slots) if not x in state]:
            state_copy = np.copy(state)
            state_copy[slot_no] = slot_i

            for state_t in self.__construct_recode_mat_aux(
                    slot_no + 1, 
                    state_copy):
                yield state_t


    def __construct_recode_mat(self):
        """
            Construct a new recoding matrix
        """
        diag_mat = np.diagflat(np.ones(shape=(self._n_slots)))
        recode_mat = np.zeros(
                shape=(self._n_states, self._n_slots, self._n_slots),
                dtype=np.float32)
        state = np.full(shape=self._n_slots, fill_value=-1, dtype=np.int)
        i = 0
        for positions in self.__construct_recode_mat_aux(0, state):
            recode_mat[i] = diag_mat[:][positions]
            i += 1
        return recode_mat
    

    def __build_graph(self):
        """
            Build TF graph for analogy-making
        """
        self.__sem_target = tf.placeholder(
                shape=[self._n_slots, self._sem_dim], 
                dtype=tf.float32)
        self.__struct_target = tf.placeholder(
                shape=[self._max_arity, self._n_slots, self._n_slots], 
                dtype=tf.float32)

        self.__sem_base = tf.placeholder(\
                shape=[None, self._n_slots, self._sem_dim], 
                dtype=tf.float32)
        self.__struct_base = tf.placeholder(
                shape=[None, self._max_arity, self._n_slots, self._n_slots], 
                dtype=tf.float32)
        
        #Construct recoding matrix
        if Path("recode/recode_mat.{}.pickle".format(self._n_slots)).is_file():
            #a recoding matrix with given parameters is already 
            # created and serialized, load it
            print("Loading recoding matix...", end="")
            sys.stdout.flush()
            with open(
                    "recode/recode_mat.{}.pickle".format(self._n_slots), 
                    "rb") as file_h:                            
                recode_mat = tf.constant(pickle.load(file_h))
            print("Done.")
            sys.stdout.flush()
        else:
            #create recoding matrix and serialize it to a file
            recode_mat = tf.constant(self.__construct_recode_mat())
            with open(
                    "recode/recode_mat.{}.pickle".format(self._n_slots), 
                    "wb") as file_h:
                with tf.Session():
                    pickle.dump(recode_mat.eval(), file_h)                    
                    
            
        #generate all possible states of the semantics of the target
        sem_targets = tf.reshape(                
                tf.matmul(
                    tf.reshape(
                        recode_mat,
                        [self._n_states * self._n_slots,self._n_slots]),
                    self.__sem_target),
                [1, self._n_states, self._n_slots * self._sem_dim])
        
        #generate all possible states of the structure of the target
        struct_targets = tf.reshape( 
            tf.transpose(
                tf.reshape(
                    tf.concat(
                        [                            
                            tf.matmul(
                                tf.reshape(
                                    tf.matmul(
                                        tf.reshape(
                                            recode_mat, 
                                            [
                                             self._n_states * self._n_slots, 
                                             self._n_slots
                                            ]), 
                                        self.__struct_target[a_i],
                                    ), 
                                    [
                                        self._n_states, 
                                        self._n_slots, 
                                        self._n_slots
                                    ]),
                                tf.transpose(recode_mat, [0, 2, 1]),
                            )
                            for a_i in range(self._max_arity)
                        ], 0
                    ),
                    [
                        self._max_arity, 
                        self._n_states, 
                        self._n_slots * self._n_slots
                    ]), 
                [1, 0, 2]),
            [
                1, 
                self._n_states, 
                self._max_arity * self._n_slots * self._n_slots
            ])
        
        #computer number of bases
        n_bases = tf.shape(self.__sem_base)[0]
        
        #reshapoe bases
        sem_base = tf.reshape(
            tf.tile(
                    tf.reshape(
                            self.__sem_base, 
                            [n_bases, self._n_slots * self._sem_dim]),
                    [1, self._n_states]),
            [n_bases, self._n_states, self._n_slots * self._sem_dim])
        
        struct_base = tf.reshape(
            tf.tile(
                    tf.reshape(
                        self.__struct_base, 
                        [
                            n_bases, 
                            self._max_arity * self._n_slots * self._n_slots
                        ]),
                    [1, self._n_states]),
            [
                n_bases, 
                self._n_states, 
                self._max_arity * self._n_slots * self._n_slots
            ])
        
        #compute semantics denominator for cosine similarity
        denom_sem = tf.multiply( 
            tf.sqrt(
                tf.reduce_sum(
                    tf.multiply(sem_targets, sem_targets), 
                    axis=[2])),                
            tf.sqrt(
                tf.reduce_sum(
                    tf.multiply(sem_base, sem_base), 
                    axis=[2])))      
        #compute numerator
        num_sem = tf.reduce_sum(tf.multiply(sem_targets, sem_base), axis=[2])        
        #compute cosine similarity
        sem_cos = tf.add(tf.multiply(tf.divide(num_sem, denom_sem), 0.5), 0.5) 
                
        #compute structure denominator for cosine similarity
        denom_struct = tf.reduce_sum(struct_targets, axis=[2])
        #compute numerator
        num_struct = tf.reduce_sum(
                tf.multiply(struct_targets, struct_base), axis=[2])
        #compute cosine similarity
        struct_cos = tf.divide(num_struct, denom_struct)
        
        similarities = tf.add(
                tf.multiply(sem_cos, 1 - self._sigma), 
                tf.multiply(struct_cos, self._sigma))

        self._sem_cos = sem_cos
        self._struct_cos = struct_cos
        
        #get maximum similarity
        base_max_similarities = tf.reduce_max(similarities, axis=[1])
        #get index of base with max similarity        
        self.__best_base_index = tf.argmax(base_max_similarities)
        #get the index of the recoding which lead to the max similarity
        best_recoding_no = \
            tf.argmax(
                tf.reshape(
                        tf.slice(
                            similarities,
                            [self.__best_base_index, 0], [1, self._n_states]),
                        [self._n_states]))
        #maximum similarity value
        self.__best_base_similarity = tf.reduce_max(base_max_similarities)                
        #best recoding
        self.__best_recoding = tf.slice(
                recode_mat, 
                [best_recoding_no, 0, 0], 
                [1, self._n_slots, self._n_slots])
        #best recoding of semantics
        self.__best_target_sem_recoding = tf.reshape(
            tf.slice(
                sem_targets, 
                [0, best_recoding_no, 0], 
                [1, 1, self._n_slots * self._sem_dim]),
            [self._n_slots, self._sem_dim])
        #best recoding of structure
        self.__best_target_struct_recoding = tf.reshape(
            tf.slice(
                    struct_targets, 
                    [0, best_recoding_no, 0], 
                    [1, 1, self._max_arity * self._n_slots * self._n_slots]
            ),
            [self._max_arity, self._n_slots, self._n_slots]
        )
        
    
    def make(self, target, bases):
        """
            Returns the predicate type id which corresponds to the given semantic vector.
            Use to recover the predicate type from a vector representation
        """      
        if not type(bases) is list:
            sem_bases = [bases[0]]
            struct_bases = [bases[1]]
        else:
            sem_bases = np.zeros(
                shape=(len(bases), self._n_slots, self._sem_dim), 
                dtype=np.float32)
            struct_bases = np.zeros(
                shape=(len(bases),
                self._max_arity, self._n_slots, self._n_slots), 
                dtype=np.float32)
            for i in range(len(bases)):
                sem_bases[i] = bases[i][0]
                struct_bases[i] = bases[i][1]
                
        with tf.Session() as session:
            results = session.run(
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
                        self.__struct_base: struct_bases
                })
        return (results[0], results[1], results[2])
        

            
    def __init__(self, sigma, n_slots, max_arity, sem_dim):
        self._sigma = sigma
        self._n_slots = n_slots
        self._max_arity = max_arity
        self._sem_dim = sem_dim
        #number of possible states of the representations
        self._n_states = int(factorial(n_slots))    
        tf.disable_eager_execution()
        self.__build_graph()
            
