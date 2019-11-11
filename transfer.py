import numpy as np
import click

class Transfer:

    def __is_mapped(self, wm_i):
        return np.sum(self._target[0][wm_i]) <> 0.0
    
    def _transfer_entity(self, base_i, args):
        self._target[0][base_i] = self._base[0][base_i]
        for a in range(args.shape[0]):
            self._target[1][a,base_i,args[a]] = 1
        self.__transferred_entities.append((base_i, args))

    def is_effective(self):
        return len(self.__transferred_entities) > 0
        
    def get_trasnferred_entities(self):
        for i in range(len(self.__transferred_entities)):
            yield self.__transferred_entities[i]
        
    def start(self):
        (sem2, struct2) = self._base

        for base_i in range(self._wm_capacity):
            if (not self.__is_mapped(base_i)) and (np.sum(struct2[0,base_i,:]) > 0):
                args = np.zeros(shape=(self._max_arity), dtype=np.int)
                args_mapped = True
                for arity in range(self._max_arity):
                    if np.sum(struct2[arity, base_i]) > 0:
                        arg_i = np.argmax(struct2[arity, base_i])
                        if not self.__is_mapped(arg_i):
                            args_mapped = False
                            break
                        else:
                            args[arity] = arg_i
                    else:
                        break
                if args_mapped:
                    self._transfer_entity(base_i, args)
                 
        return self._target    
            
    def __init__(self, target, base):
        self._target = target
        self._base = base
        self.__transferred_entities = []
        (self._max_arity, self._wm_capacity, _) = base[1].shape
        
class SimilarityTresholdProbabilisticTransfer(Transfer):
    def _transfer_entity(self, base_i, args):
        if (np.random.rand() < self.__similarity):
            return Transfer._transfer_entity(self, base_i, args)
        
    def __init__(self, target, base, similarity):
        Transfer.__init__(self, target, base)
        self.__similarity = similarity
        
class InteractiveTransfer(Transfer):
    def _transfer_entity(self, base_i, args):
        
        if (click.confirm("Transfer {0} to {1}-?({2})".format(
                str(self.__base_predicates[base_i]), 
                str(self.__base_predicates[base_i].get_type_id()),
                ", ".join(
                    [str(self.__target_predicates[arg]) for arg in args]
                )
            ), 
            default=True)):
            return Transfer._transfer_entity(self, base_i, args)
            
    def __init__(self, target, base, target_predicates, base_predicates):
        Transfer.__init__(self, target, base)
        self.__target_predicates = target_predicates
        self.__base_predicates = base_predicates
        
class BayesianTransfer(Transfer):
    def _transfer_entity(self, base_i, args):
        #TODO
        raise BaseException("Not implemented")
    