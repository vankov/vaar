import numpy as np
import tensorflow as tf

from predicates import Atom, Predicate
from encoding import VectorEncoder, Glove50Encoder
from tools import print_v

from  analogy import VectorAnalogy 
from comparison import VectorMapping

N_SLOTS = 6
MAX_ARITY = 2
encoder = Glove50Encoder(N_SLOTS, MAX_ARITY)    
            
john_1 = Atom("John")  
mary_1 = Atom("Mary")
john_loves_mary = Predicate("loves", 2, [john_1], [mary_1])

john_2 = Atom("John")  
mary_2 = Atom("Mary")
mary_loves_john = Predicate("loves", 2, [mary_2], [john_2])

john_3 = Atom("John")  
mary_3 = Atom("Mary")
john_hates_mary = Predicate("talks", 2, [john_3], [mary_3])

john_4 = Atom("John")  
mary_4 = Atom("Mary")
mary_hates_john = Predicate("hates", 2, [mary_4], [john_4])

peter = Atom("Peter")  
helen = Atom("Helen")
peter_loves_helen = Predicate("loves", 2, [peter], [helen])

john_5 = Atom("John")
mary_5 = Atom("Mary")

target = john_loves_mary    

bases_predicates = [
#        [john_loves_mary],
        [mary_loves_john],
        [john_hates_mary],
        [mary_hates_john],
        [peter_loves_helen],
        [john_5, mary_5]
    ]

bases_v = []
bases = {}
base_ps = {}
for i in range(len(bases_predicates)):
    vec_repr, ps = encoder.encode_predicates(bases_predicates[i])
    bases[i] = vec_repr
    bases_v.append(vec_repr)
    base_ps[i] = ps

target_v, target_ps = encoder.encode_predicates([target])    

def prepare_vars(vars_r):
    
    n_vars_r = np.zeros(
            shape=(
                N_SLOTS, 
                encoder.get_sem_dim() + MAX_ARITY * N_SLOTS))
    
    sem, struct = vars_r
    
    for i in range(N_SLOTS):
        n_vars_r[i,:encoder.get_sem_dim()] = sem[i]
        for a in range(MAX_ARITY):
            n_vars_r[
                i,
                encoder.get_sem_dim() + a * N_SLOTS
                :encoder.get_sem_dim() + (a + 1) * N_SLOTS] = struct[a][i]
    
    return n_vars_r.flatten()

VARS_TOTAL_DIM = N_SLOTS * (encoder.get_sem_dim() + MAX_ARITY * N_SLOTS)

#test_set1 = np.zeros(shape=(len(bases) + 1, VARS_TOTAL_DIM, 2))
test_set2 = np.zeros(shape=(len(bases), VARS_TOTAL_DIM, 2))
#test_set1[:,:,0] = prepare_vars(target_v)
test_set2[:,:,0] = prepare_vars(target_v)

#test_set1[len(bases),:,1] = prepare_vars(target_v)

for i in range(len(bases_predicates)):
#    test_set1[i,:,1] = prepare_vars(bases_v[i])
    test_set2[i,:,1] = prepare_vars(bases_v[i])
    
#np.save("test1", test_set1)
np.save("test2", test_set2)
#t_sem = target_v[0].flatten()
#np.random.shuffle(t_sem)
#t_struct = target_v[1].flatten()
#np.random.shuffle(t_struct)
#
#target_v = (
#    t_sem.reshape(N_SLOTS, encoder.get_sem_dim()),
#    t_struct.reshape(MAX_ARITY, N_SLOTS, N_SLOTS))


#print_v(target_v, precision=1)

analogy = VectorAnalogy(0.5, N_SLOTS, MAX_ARITY, encoder.get_sem_dim())    
vm = VectorMapping(0.5)

for base_i in bases:
    (max_sim, best_base_i, mapping) = analogy.make(target_v, bases_v[base_i])    
    
    best_base_i = base_i
    #print_v "None"
#    print([str(p) for p in target_ps])(mapping)
#            print_v(target)
#            print_v(bases[base_i])
#            print(mapping)
#            john_mapping =
    for i in range(len(base_ps[best_base_i])):
        base_p = base_ps[best_base_i][i]
        target_p = target_ps[np.argmax(mapping[:, i])]
        print("{}<->{}".format(target_p, base_p))

    
    print("base: {}, sim: {:.5}, p3 sim: {:.5}".format(
            best_base_i, 
            max_sim,
            vm.map_v(target_v, bases_v[base_i])[0]
          ))
    print("")
#    exit(0)