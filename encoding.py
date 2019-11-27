import numpy as np

from semantics import Glove50Semantics, Glove100Semantics
from semantics import Glove200Semantics, Glove300Semantics
from semantics import KazumaSemantics

class VectorEncoder:
    
    def get_sem_dim(self):
        """
            Returns the dimension of the semantic vectors
        """
        return self.__sem_dim
        
    def _encode_predicate_semantics(self, predicate):
        """
            Returns a vector of frandom numbers
            Supposed to be overriden in order to use plausible semantic representations
        """        
        p_type_id = predicate.get_type_id()
        if not (p_type_id in self.__predicate_encodings):
            self.__predicate_encodings[p_type_id] = np.random.uniform(size=(self.__sem_dim))
                    
        return self.__predicate_encodings[p_type_id]            
    
    def __get_free_index(self):
        """
            Returns an available representational index.
        """
        if len(self.__free_indicies) < 1:
            raise ("Out of representational indices!")
        
        return self.__free_indicies.pop()            
   
    def __encode_predicate_aux(self, predicate, vec_repr):
        """
            Encodes a predicate. It also encodes all the predicates which the
                variables of the current predicate are bound to
        """
        for slot_i in range(self.__n_slots):
            if (predicate.same_as(self.__encoded_predicates[slot_i])):
                return slot_i

        free_slot_i = self.__get_free_index()
        self.__encoded_predicates[free_slot_i] = predicate
        vec_repr[0][free_slot_i] = self._encode_predicate_semantics(predicate)
        
        args = predicate.get_arguments()        

        for arg_no in range(len(args)):
            #Check if the argument is bound
            if (not (args[arg_no] == None)):
                for var_binding_i in range(len(args[arg_no])):
                    arg_index = self.__encode_predicate_aux(args[arg_no][var_binding_i], vec_repr)
                    #Binds argument
                    vec_repr[1][arg_no, free_slot_i, arg_index] = 1
            else:
                raise ("Can't encode an unresolved predicate!")
                
        return free_slot_i
    
    def _shuffle_indicies(self):
        """
            Can be overriden in order to implement pseudo-random representation index allocation
        """
        np.random.shuffle(self.__free_indicies)
        
    def encode_predicates(self, predicates):
        semantics = np.random.normal(
                loc=0,
                scale=0,
                size=(self.__n_slots, self.__sem_dim))
        structure = np.ones(
                shape=(self.__max_arity, self.__n_slots, self.__n_slots), 
                dtype=self.__dtype) * -1.0
        assert type(predicates) is list
        
        self.__encoded_predicates = []
        self.__free_indicies = list(range(self.__n_slots))
        
        self._shuffle_indicies()
        
        for i in range(self.__n_slots):
            self.__encoded_predicates.append(None)
            
        for i in range(len(predicates)):
            self.__encode_predicate_aux(predicates[i], (semantics, structure))
            
        return (semantics, structure), self.__encoded_predicates
    
    def __init__(self, n_slots, max_arity, sem_dim, dtype = np.float32):
        self.__n_slots = n_slots
        self.__max_arity = max_arity
        self.__sem_dim = sem_dim
        self.__dtype = dtype
        self.__predicate_encodings = {}
        
class Glove50Encoder(VectorEncoder):
    
    def _encode_predicate_semantics(self, predicate):
        return self.__ltm.get_semantic_vector(str(predicate.get_type_id()), True)
        
    def __init__(self, n_slots, max_arity, dtype = np.float32):
        VectorEncoder.__init__(self, n_slots, max_arity, 50, dtype = np.float32)
        self.__ltm = Glove50Semantics()

class Glove100Encoder(VectorEncoder):
    
    def _encode_predicate_semantics(self, predicate):
        return self.__ltm.get_semantic_vector(str(predicate.get_type_id()), True)
        
    def __init__(self, n_slots, max_arity, dtype = np.float32):
        VectorEncoder.__init__(self, n_slots, max_arity, 100, dtype = np.float32)
        self.__ltm = Glove100Semantics()

class Glove200Encoder(VectorEncoder):
    
    def _encode_predicate_semantics(self, predicate):
        return self.__ltm.get_semantic_vector(str(predicate.get_type_id()), True)
        
    def __init__(self, n_slots, max_arity, dtype = np.float32):
        VectorEncoder.__init__(self, n_slots, max_arity, 200, dtype = np.float32)
        self.__ltm = Glove200Semantics()


class Glove300Encoder(VectorEncoder):
    
    def _encode_predicate_semantics(self, predicate):
        return self.__ltm.get_semantic_vector(str(predicate.get_type_id()), True)
        
    def __init__(self, n_slots, max_arity, dtype = np.float32):
        VectorEncoder.__init__(self, n_slots, max_arity, 300, dtype = np.float32)
        self.__ltm = Glove300Semantics()

class KazumaEncoder(VectorEncoder):
    
    def _encode_predicate_semantics(self, predicate):
        return self.__ltm.get_semantic_vector(str(predicate.get_type_id()), True)
        
    def __init__(self, n_slots, max_arity, dtype = np.float32):
        VectorEncoder.__init__(self, n_slots, max_arity, 100, dtype = np.float32)
        self.__ltm = KazumaSemantics()
