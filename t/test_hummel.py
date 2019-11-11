from analogy import NeuralAnalogy
import numpy as np

from predicates import Atom, Predicate
from encoding import Glove50Encoder

#number of representational slots
r_slots_n = 9
#semantics dimensionality (number of units)
sem_dim = 50
#maximal arity
max_ar = 2
#tha relative weight of semantics and structure
sigma = 0.5#semantics and structure have equal weight

#target situation
john_t = Atom("John")     
mary_t = Atom("Mary")
apple_t = Atom("apple")
core_t = Atom("core")
bill_t = Atom("Bill")     
loves_t = Predicate("loves", 2, [john_t], [mary_t])
has_t = Predicate("has", 2, [apple_t], [core_t])
knows1_t = Predicate("knows", 2, [bill_t], [has_t])
knows2_t = Predicate("knows", 2, [john_t], [loves_t])


#base
john_b1 = Atom("John")     
mary_b1 = Atom("Mary")
bill_b1 = Atom("Bill")
apple_b1 = Atom("apple")
core_b1 = Atom("core")
loves_b1 = Predicate("loves", 2, [john_b1], [mary_b1])
has_b1 = Predicate("has", 2, [apple_b1], [core_b1])
knows1_b1 = Predicate("knows", 2, [bill_b1], [loves_b1])
knows2_b1 = Predicate("knows", 2, [john_b1], [has_b1])


#Vector encoder using Glove 50d word word embeddings
encoder = Glove50Encoder(r_slots_n, max_ar)    
#encoder = VectorEncoder(r_slots_n, max_ar, sem_dim)    

#encode target
#target_v is a vector represenbtation, target_ps contains the corresponding predicatres
target_v, target_ps = encoder.encode_predicates([loves_t, knows1_t, knows2_t])
#encode base
base_v, base_p  = encoder.encode_predicates([loves_b1, knows1_b1, knows2_b1])
    
#create neuual analogical machine
neuro_analogy = NeuralAnalogy(sigma, r_slots_n, max_ar, sem_dim)            
#run mapping
(max_sim, _, mapping) = neuro_analogy.make(target_v, base_v)    

#Print the overal simialriyt of the target and the basae
print("Similarity: {}".format(max_sim))

#Print mappings
for i in range(r_slots_n):
    target_p = target_ps[np.argmax(mapping[:, i])]
    print("{}<->{}".format(target_p, base_p[i]))
