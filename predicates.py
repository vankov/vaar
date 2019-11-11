class Variable(object):
    """
        A predicate calculus variable
    """    
    def get_value(self):
        """
            Returns the value the variable is bound to
        """
        return self.__value
    
    def is_bound(self):
        """
            Checks whether the variable is bound
        """        
        return self.get_value() != None
    
    def bind(self, value):
        """
            Binds a variable to a value (i.e. to a predicate)
        """                
        self.__value = value
            
    def __str__(self):
        """
            Returns a string representation of the variable. If the variable is not bound, the returned value is "_"
        """
        if self.is_bound():
            value = self.get_value()
            if type(value) is list:
                if (len(value) == 1):
                    return str(value[0])
                else:
                    return "({})".format(", ".join([str(x) for x in self.get_value()]))
            else:
                return str(self.get_value())
        else:
            return "_"
            
    def __init__(self):
        """
            Creates a new variable
        """
        self.__value = None

class Predicate(object):
    """
        Representation of a predicate
    """
    
    __predicate_instaces = {}   #maps predicate types to predicate instances
       
    def bind(self, *values):
        """
            Binds the arguments(variables) of the predicate to a list of values.
            If the arity of the predicate is larger than the number of values passed (n), then only the first 'n' vriables are bound 
            If the number of values is longer than the arity of the predicate, then only the first 'arity' values are used 
        """
        for arg_no in range(min([self.get_arity(), len(list(values))])):
            if (not self.__vars[arg_no].is_bound()):
                self.__vars[arg_no].bind(values[arg_no])
            else:
                raise BaseException("argument no {0} of {1} is already bound to {2}".format(
                        arg_no, 
                        self.get_id(), 
                        self.__vars[arg_no].get_value()
                    ))
        
    def bind_argument(self, arg_no, value):
        """
            Binds the 'arg_no'-th argument of the predicate to the specified value
        """
        self.__vars[arg_no].bind(value)
        
    def get_arity(self):
        """
            Returns the arity of the predicate
        """
        return self.__arity
    
    def get_order(self):           
        """
            Returns the order of the predicate.
            The arguments of the predicate must be bound to computer order 
        """
        order = 0
        
        for i in range(len(self.__vars)):
            if (self.__vars[i].is_bound()):
                order = max([order, 1 + self.__vars[i].get_value().get_order()])
        
        return order
    
    def get_id(self):
        """
            Returns an unique string identifying the predicate instance (i.e. the token)
            For example:
                cat-0, cat-1, chase-0
        """
        return str(self.__type_id) + "-" + str(self.__instance_no)

    def get_type_id(self):
        """
            Returns the id of the predicate type.
            For example:
                cat, dog, chase
        """
        return self.__type_id
    
    def is_resolved(self):     
        """
            Checks whether all the variables of the predicate are bound
        """   
        for i in range(len(self.__vars)):
            if (not(self.__vars[i].is_bound() and self.__vars[i].get_value().is_resolved())): 
                return False
        
        return True
    
    def same_as(self, predicate):
        """
            Checks whether the predicate is identical to another predicate
        """
        if (predicate == None):
            return False
        return self.get_id() == predicate.get_id() 
    
    def get_arguments(self):
        """
            Returns a list of predicates, which the variables of the predicate are bound to.
            If a variable is not bound, the corresponding element in the list will be None
        """
        args = []
        for i in range(len(self.__vars)):
            if (self.__vars[i].is_bound()):
                args.append(self.__vars[i].get_value())
            else:
                args.append(None)
    
        return args
    
    def get_semantic_vector(self, sltm):
        """
            Returns a semantic vector representation of the predicate.
            All instances of the same predicate type have the same semantic vector  
        """
        return sltm.get_semantic_vector(self.__type_id, True)
    
    def __str__(self):
        """
            Returns a string representation of the predicate and its arguments.
            For example:
                cat-3, dog-1, chase-0(dog-1, cat-3)
        """
        if (self.get_arity() > 0):
            args_str = "({0})".format(",".join(map(str, self.__vars)))
        else:
            args_str = ""
        
        return "{0}{1}".format(self.get_id(), args_str)
        
    def __init__(self, type_id, arity, *values):
        """
            Creates a new predicate of the specified type and arity. 
            The variables of the predicate are bound to the specified values 
        """                
        if not type_id in self.__predicate_instaces:
            self.__predicate_instaces[type_id] = []
          
        self.__type_id = type_id
        self.__instance_no = len(self.__predicate_instaces[type_id])

        self.__predicate_instaces[type_id].append(self)
        self.__arity = arity
        self.__vars = []
        
        for _ in range(arity):
            self.__vars.append(Variable()) 
            
        self.bind(*values)

class Atom(Predicate):
    def __init__(self, type_id):
        Predicate.__init__(self, type_id, 0)
        
class Property(Predicate):
    def __init__(self, type_id, value = None):
        if (value == None):
            Predicate.__init__(self, type_id, 1)
        else:
            Predicate.__init__(self, type_id, 1, value)
    
class Episode(Predicate):
    def __init__(self, *predicates):
        Predicate.__init__(self, "Episode", len(predicates),*predicates)