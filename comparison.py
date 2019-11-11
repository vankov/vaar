from __future__ import print_function

import numpy as np
from scipy import spatial


class VectorMapping:
           
    def __get_encoding_aux(self, n_predicates, n_slots, predicate_no, state):
            
        if (predicate_no >= n_predicates):
            yield state
            return
        
        for t in [x for x in range(n_slots) if not (x in state)]:
            state_copy = np.copy(state)
            state_copy[predicate_no] = t
       
            for state_t in self.__get_encoding_aux(
                    n_predicates, n_slots, predicate_no + 1, state_copy):
                yield state_t

        
    def __get_encoding(self, n_predicates, n_slots):
        state = np.full(shape=n_predicates, fill_value=-1, dtype=np.int)
        return self.__get_encoding_aux(n_predicates, n_slots, 0, state)
        
    def _compare_semantics(self, sem_v1, sem_v2):
        
        denom = \
            spatial.distance.norm(sem_v1[0]) \
            * \
            spatial.distance.norm(sem_v2[0])
        
        if (denom == 0):
            return 0
        else:
            return 0.5 + (
                    np.dot(
                        sem_v1[0].flatten(), 
                        sem_v2[0].flatten()
                    ) / denom) / 2
    
    def _compare_structure(self, struct_v1, struct_v2):
        
        denom = \
            spatial.distance.norm(struct_v1[1]) \
            * \
            spatial.distance.norm(struct_v2[1])
            
        if (denom == 0):            
            return 0
        else:            
            return (np.dot(struct_v1[1].flatten(), struct_v2[1].flatten()) 
                    / denom)
        
        
    def __recode_representation(self, representation, encoding):
        semantics, structure = representation
        new_semantics = np.zeros_like (semantics)
        new_structure = np.zeros_like(structure)
        
        (max_arity, n_slots, _) = structure.shape
        bindings = {}
        p_no = 0
        
        for t in range(n_slots):
            if (np.sum(semantics[t]) != 0.0):
                new_t = encoding[p_no]
                new_semantics[new_t] = semantics[t]
                bindings[t] = new_t
                p_no += 1
                
        for arg_no in range(max_arity):
            for higher in bindings:
                for lower in bindings:
                    new_structure[
                        arg_no, 
                        bindings[higher], 
                        bindings[lower]] = structure[arg_no, higher, lower]
        
        return (new_semantics, new_structure)
        
    def map_v(self, v1, v2):
        
        assert(v1[0].shape == v2[0].shape)
        assert(v1[1].shape == v2[1].shape)
        
        
        (max_arity, n_slots, _) = v1[1].shape        
        n_predicates = np.sum([1 for x in np.sum(v2[0], axis=(1)) if x != 0.0])
        
        max_similarity = 0
        most_similar_v2 = v2
        
        for representation in self.__get_wm_state(n_predicates, n_slots):
            v2_t = self.__recode_representation(v2, representation)
            
            t_sim = \
                    (1 - self._sigma) * self._compare_semantics(v1, v2_t) \
                    + \
                    self._sigma * self._compare_structure(v1, v2_t)
                    
            if (t_sim >= max_similarity):
                max_similarity = t_sim
                most_similar_v2 = v2_t
            
        return (max_similarity, most_similar_v2)
        
    def __init__(self, systematicity):
        self._sigma = systematicity
        
class FastSemanticMapping(VectorMapping):
    
    def compare(self, r1, r2):
        (sem_v1, bind_v1) = r1
        (sem_v2, bind_v2) = r2
                
        assert(sem_v1.shape == sem_v2.shape)
        
        
        return (self._compare_semantics(
                (np.sum(sem_v1, axis=(0)), bind_v1), 
                (np.sum(sem_v2, axis=(0)), bind_v2)
            ), r2)
        
    def __init__(self):
        VectorMapping.__init__(self, 0)
