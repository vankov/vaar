import numpy as np
from scipy import spatial

class Semantics:
    """
        Vector based semantic Long Term Memory. 
    """
    __type_ids = []
    #A dictionary mapping semantic represenations (semantic vectors) to predicate ids
    __semantics = {}
    #A dictionary mapping predicate ids to semantic vectors
    __semantics_categories = {}
    
    def _get_semantic_id(self, vector):
        
        return "".join(list(map(str, vector)))
        
    def get_semantic_category(self, semantic_vector):
        """
            Returns the predicate type id which corresponds to the given semantic vector.
            Use to recover the predicate type from a vector representation
        """         
        
        assert(semantic_vector.shape[0] == self.get_semantic_dimension())
        
        sem_v_id = self._get_semantic_id(semantic_vector)
        
        if sem_v_id in self.__semantics:
            return self.__semantics[sem_v_id]
        else:
            return None
            
    def add_category(self, predicate_type_id, semantic_vector):
        """
            Creates a new semantic category or updates an existing one
        """
        assert(semantic_vector.shape[0] == self.get_semantic_dimension())
        
        sem_v_id = self._get_semantic_id(semantic_vector)
        
        self.__semantics[sem_v_id] = predicate_type_id
        self.__semantics_categories[predicate_type_id] = semantic_vector
        
        
    def get_semantic_dimension(self):
        """
            Returns the dimension of the semantic space
        """
        return self.__sem_dim
    
    def _populate_semantic_vector(self, predicate_type_id):
        return np.random.uniform(size=self.get_semantic_dimension(), low=-1)
        
    def get_semantic_vector(self, predicate_type_id, populate = False):
        """
            Returns the semantic vector representation of a category identified by a predicate type id
            If the category is not represented in current semantic LTM  and the flag 
                'populate' is turned on, then the category is created and its representation
                is populated with a random semantic vector
        """
        if predicate_type_id in self.__semantics_categories:
            return self.__semantics_categories[predicate_type_id]
    
        if populate:           
            vector = self._populate_semantic_vector(predicate_type_id)             
            self.add_category(
                predicate_type_id,
                vector
            )
            return vector
            
        else:
            return None            
    
    def __init__(self, sem_dim):
        """
            Constructs a new semantic Long Term Memory with the specified dimension of the semantic space
        """
        self.__sem_dim = sem_dim
         
class LocalistSemantics(Semantics):
    predicate_type_id_localist_units = {}
    max_unit = 0
    
    def _populate_semantic_vector(self, predicate_type_id):
        v = np.zeros(shape=(self.get_semantic_dimension()))
        if predicate_type_id not in self.predicate_type_id_localist_units:
            assert(self.max_unit < self.get_semantic_dimension())
            self.predicate_type_id_localist_units[predicate_type_id] = self.max_unit
            self.max_unit += 1
        v[self.predicate_type_id_localist_units[predicate_type_id]] = 1
        return v
        
    def __init__(self, semantic_dim):        
        Semantics.__init__(self, semantic_dim)
               
class WordEmbeddingsSemantics(Semantics):
    
    def _populate_semantic_vector(self, predicate_type_id):
        predicate_type_id_str = str(predicate_type_id)
        with open(self.__words_embeddings_file, 'r') as embeddings_f:
            for line in embeddings_f:
                row = line.strip().split(' ')
                if row[0].lower() == predicate_type_id_str.lower():
                    return np.array(list(map(float, row[1:])))
                    
        print("Can't find embedding for {}".format(predicate_type_id_str))
        return Semantics._populate_semantic_vector(self, predicate_type_id)                
    
    def __init__(self, words_embeddings_file, semantic_dim):        
        self.__words_embeddings_file = words_embeddings_file
        Semantics.__init__(self, semantic_dim)

    
class KazumaSemantics(WordEmbeddingsSemantics):    
    def __init__(self):        
        WordEmbeddingsSemantics.__init__(self, "kazuma.embeddings.txt", 100)
    
                
class Glove50Semantics(WordEmbeddingsSemantics):        
    def __init__(self):                
        WordEmbeddingsSemantics.__init__(self, "glove.50d.embeddings.txt", 50)

class Glove100Semantics(WordEmbeddingsSemantics):        
    def __init__(self):                
        WordEmbeddingsSemantics.__init__(self, "glove.100d.embeddings.txt", 100)        
        
class Glove200Semantics(WordEmbeddingsSemantics):        
    def __init__(self):                
        WordEmbeddingsSemantics.__init__(self, "glove.200d.embeddings.txt", 200)        
        
class Glove300Semantics(WordEmbeddingsSemantics):        
    def __init__(self):                
        WordEmbeddingsSemantics.__init__(self, "glove.300d.embeddings.txt", 300)        
