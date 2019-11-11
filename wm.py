import numpy as np
from predicates import Predicate

class WM:    
    """
        Working Memory
        Used to store activated predicates and their variable bindings. 
        Supports both predicate and vector based representations
    """
    def get_vector_representation(self, sltm):
        """
            Returns a vector representations of the contents of the working memory
            The vector representations consists of two parts:
                semantics - the semantics vectors of the activated predicates
                    the dimension of 'semantics' is wm_capacity x semantic_dimension, 
                    where semantic dimension is the dimension of the semantic space and 
                    wm_capacity is the working memory capacity 
                structure - binary vectors (0 and 1) describing the variable bindings
                    the dimension of structure is max_arity x wm_capacity x wm_capacity, 
                    where max_arity is the maximum predicate arity which can be represented
                    and wm_capicity is the working memory capacity
                      
        """
        semantics = np.zeros(shape=(self.__capacity, sltm.get_semantic_dimension()))

        for wm_i in range(len(self.__contents)):
            if (self.__contents[wm_i] <> None):
                semantics[wm_i] = self.__contents[wm_i].get_semantic_vector(sltm)
                 
        return (semantics, self.__structure)
    
    def get_predicate_representation(self, (semantics, structure), sltm):
        """
            Returns a lift of predicate representations of the contents of the working memory 
        """        
        (wm_capacity, _) = semantics.shape
        arities = np.sum(structure <> 0, axis=(0, 2))
                
        ps = {}
                
        for sem_t in range(wm_capacity):
            if (np.sum(semantics[sem_t]) <> 0):
                p_type_id = sltm.get_semantic_category(semantics[sem_t])

                ps[sem_t] = Predicate(p_type_id, arities[sem_t])
                
        for sem_t, p in ps.iteritems():
            if (not p.is_resolved()):
                for a_no in range(p.get_arity()):
                    #Python and NumPy are beautiful
                    p.bind_argument(a_no, ps[np.argmax(structure[a_no, sem_t])])
            
        return ps
    
    def __get_free_index(self):
        """
            Returns an available working memory index.
        """
        wm_i = None
        while True:
            wm_i = np.random.randint(0, self.__capacity)
            if (self.__contents[wm_i] == None):
                break
        return wm_i
    
    def __bind_argument(self, predicate_index, argument_no, argument_index):
        """
            Binds a predicate argument (specified by the id of the predicate instance the number of the argument) 
                to a predicate with the specified id (argument_index) 
        """
        self.__structure[argument_no, predicate_index, argument_index] = 1
        
    def add_predicate(self, predicate):
        """
            Adds a predicate to working memory. It also adds all the predicates which the
                variables of the current predicate are bound to
        """
        for wm_i in range(self.__capacity):
            if (predicate.same_as(self.__contents[wm_i])):
                return wm_i

        free_wm_i = self.__get_free_index()
        self.__contents[free_wm_i] = predicate
            
        args = predicate.get_arguments()        

        for arg_no in range(len(args)):
            if (not (args[arg_no] == None)):
                for var_binding_i in range(len(args[arg_no])):
                    arg_index = self.add_predicate(args[arg_no][var_binding_i])
                    self.__bind_argument(free_wm_i, arg_no, arg_index)
            else:
                raise ("Can't add an unresolved predicate to WM")
                
        return free_wm_i
    
    def reset(self):
        for wm_i in range(self.__capacity):
            self.__contents[wm_i] = None
        self.__structure = np.zeros_like(self.__structure)
        
    def __str__(self):
        """
            Returns a string predicate representation of the predicates in working memory and their bindings 
        """
        return "\n".join(map(str, self.__contents))
    
    def __init__(self, capacity, max_arity):
        """
            Constructs a new instance of working memory
            capacity is the the maximum number of predicates which can be represented
            max_arity is the maximum arity of the predicated which can be represented
        """
        self.__contents = []
        self.__capacity = capacity
        for _ in range(capacity):
            self.__contents.append(None)
        self.__structure = np.zeros(shape=(max_arity, capacity, capacity), dtype=np.int0)
    