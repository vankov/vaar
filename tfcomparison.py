import tensorflow as tf
import numpy as np
from scipy import misc as spm
import pickle
from pathlib import Path

from comparison import Comparison

class TFComparison(Comparison):
    
    def __construct_search_mat_aux(self, wm_i, state):
            
        if (wm_i >= self.__wm_c):
            yield(state)
            return
        
        for t in [x for x in range(self.__wm_c) if not (x in state)]:
            state_t1 = np.copy(state)
            state_t1[wm_i] = t
       
            for state_t in self.__construct_search_mat_aux(wm_i + 1, state_t1):
                yield state_t
    
        
    def __construct_search_mat(self):
        diag_mat = np.diagflat(np.ones(shape=(self.__wm_c)))
        search_mat = np.zeros(shape=(self.__n_search, self.__wm_c, self.__wm_c), dtype=np.float32)
        state = np.full(shape=self.__wm_c, fill_value=-1, dtype=np.int)
        i = 0
        for x in self.__construct_search_mat_aux(0, state):
            search_mat[i] = diag_mat[:][x]
            i += 1
        return search_mat
    
    
    def __create_graph(self):
        self.__sem_input = tf.placeholder(shape=[self.__wm_c, self.__sem_dim], dtype=tf.float32)
        self.__struct_input = tf.placeholder(shape=[self.__max_ar, self.__wm_c, self.__wm_c], dtype=tf.float32)
 
        self.__sem_target = tf.placeholder(shape=[self.__wm_c, self.__sem_dim], dtype=tf.float32)
        self.__struct_target = tf.placeholder(shape=[self.__max_ar, self.__wm_c, self.__wm_c], dtype=tf.float32)
        
        if Path("search/search_mat.{}.{}.pickle".format(self.__wm_c, self.__max_ar)).is_file():
            with open("search/search_mat.{}.{}.pickle".format(self.__wm_c, self.__max_ar), "r") as f:            
                search_mat = tf.constant(pickle.load(f))
        else:
            search_mat = tf.constant(self.__construct_search_mat())
            with open("search/search_mat.{}.{}.pickle".format(self.__wm_c, self.__max_ar), "w") as f:
                pickle.dump(search_mat.eval(), f)
            
        sem_outputs = tf.reshape(
                tf.matmul(tf.reshape(search_mat, [self.__n_search * self.__wm_c,self.__wm_c]), self.__sem_input),
                [self.__n_search, self.__wm_c * self.__sem_dim]
            )
        
        struct_outputs = tf.reshape( 
            tf.transpose(
                tf.reshape(
                    tf.concat(
                        [                            
                            tf.matmul(
                                tf.reshape(
                                    tf.matmul(
                                        tf.reshape(search_mat, [self.__n_search * self.__wm_c, self.__wm_c]), 
                                        self.__struct_input[a_i],
                                    ), 
                                    [self.__n_search, self.__wm_c, self.__wm_c]
                                ),
                                tf.transpose(search_mat, [0, 2, 1]),
                            )
                            for a_i in range(self.__max_ar)
                        ], 0
                    ), [self.__max_ar, self.__n_search, self.__wm_c * self.__wm_c]
                 ), 
                [1, 0, 2])
            ,
            [self.__n_search, self.__max_ar * self.__wm_c * self.__wm_c])
        
        sem_target = tf.reshape(
            tf.tile(tf.reshape(self.__sem_target, [self.__wm_c * self.__sem_dim]),[self.__n_search]),
            [self.__n_search, self.__wm_c * self.__sem_dim])
        
        struct_target = tf.reshape(
            tf.tile(tf.reshape(self.__struct_target, [self.__max_ar * self.__wm_c * self.__wm_c]),[self.__n_search]),
            [self.__n_search, self.__max_ar * self.__wm_c * self.__wm_c])
        
        denom_sem = tf.multiply( 
            tf.sqrt(tf.reduce_sum(tf.multiply(sem_outputs, sem_outputs), axis=[1])),                
            tf.sqrt(tf.reduce_sum(tf.multiply(sem_target, sem_target), axis=[1]))
        )        
        num_sem = tf.reduce_sum(tf.multiply(sem_outputs, sem_target), axis=[1])        
        sem_cos = tf.add(tf.multiply(tf.divide(num_sem, denom_sem), 0.5), 0.5) 
        
        denom_struct = tf.multiply( 
            tf.sqrt(tf.reduce_sum(tf.multiply(struct_outputs, struct_outputs), axis=[1])),                
            tf.sqrt(tf.reduce_sum(tf.multiply(struct_target, struct_target), axis=[1]))
        )        
        num_struct = tf.reduce_sum(tf.multiply(struct_outputs, struct_target), axis=[1])        
        struct_cos = tf.divide(num_struct, denom_struct)
        
        simililarities = tf.add(tf.multiply(sem_cos, 1 - self._sigma), tf.multiply(struct_cos, self._sigma))
          
        self.__max_sim = tf.reduce_max(simililarities)
        self.__best_representation_sem = tf.reshape( 
                tf.slice(sem_outputs, [tf.argmax(simililarities), 0], [1, self.__wm_c * self.__sem_dim]),
                [self.__wm_c, self.__sem_dim] 
            )
        self.__best_representation_struct = tf.reshape( 
                tf.slice(struct_outputs, [tf.argmax(simililarities), 0], [1, self.__max_ar * self.__wm_c * self.__wm_c]),
                [self.__max_ar, self.__wm_c, self.__wm_c] 
            )
        
        #norm = tf.
        #sem_coms = 
        
    def compare(self, r1, r2):
        with tf.Session() as session:
            results = session.run(
                [self.__max_sim, self.__best_representation_sem, self.__best_representation_struct],
                feed_dict={
                    self.__sem_input: r1[0],
                    self.__struct_input: r1[1],
                    self.__sem_target: r2[0],
                    self.__struct_target: r2[1] 
                })
            return (results[0], (results[1], results[2]))
    
    def __init__(self, sigma, wm_c, max_ar, sem_dim):
        Comparison.__init__(self, sigma)
        self.__wm_c = wm_c
        self.__max_ar = max_ar
        self.__sem_dim = sem_dim
        self.__n_search = int(spm.factorial(wm_c))
        self.__create_graph()
    
    
# wm_c = 8
#     
# sltm = KazumaSemanticLTM()
# 
# house = Atom("house")    
# man = Atom("man")
# brown_house = Property("brown", [house])
# lives_in1 = Predicate("lives-in", 2, [man], [house])
# ball = Atom("ball")
# yellow_ball = Property("yellow", [ball])
# large_ball = Property("large", [ball])
# has = Predicate("has",2, [man], [ball, house])
# 
# wm = WM(wm_c, 2)
# wm.add_predicate(brown_house)
# wm.add_predicate(lives_in1)
# wm.add_predicate(yellow_ball)
# wm.add_predicate(large_ball)
# wm.add_predicate(has)
# v_repr1 = wm.get_vector_representation(sltm)
# 
# wm.reset()
# kennel = Atom("kennel")
# dog = Atom("dog")
# black_kennel = Property("black", [kennel])
# lives_in2 = Predicate("lives-in", 2, [dog], [kennel])
# wm.add_predicate(black_kennel)
# wm.add_predicate(lives_in2)
# v_repr2 = wm.get_vector_representation(sltm)
# 
# 
# def benchmark():
#     cmps = [
#         Comparison(0.9),
#         TFComparison(0.9, wm_c, 2, sltm.get_semantic_dimension())]
#     cmp_strs = [
#         "Sequential", 
#         "ParallelTF"
#     ]
#     
#     for i in range(len(cmps)):
#         t0 = time.time()
#         (max_sim, best_repr) = cmps[i].compare(v_repr2, v_repr1)
#         dt = time.time() - t0
#         print("{:^12}\t{:^5.2f}\t{:^10.4f}".format(cmp_strs[i], max_sim, dt))
# benchmark()

