from analogy import NeuralAnalogy
import numpy as np

from predicates import Atom, Predicate
from encoding import Glove50Encoder, Glove100Encoder, KazumaEncoder, VectorEncoder

from tools import print_v

#number of representational slots
r_slots_n = 8
#semantics dimensionality (number of units)
sem_dim = 50
#maximal arity
max_ar = 2
#tha relative weight of semantics and structure
sigma = 0.5#semantics and structure have equal weight

#target situation
bill_t = Atom("Bill")
joe_t = Atom("Joe")     
susan_t = Atom("Susan")
pear_t = Atom("pear")
peach_t = Atom("peach")
loves_t = Predicate("loves", 2, [bill_t], [susan_t])
has1_t = Predicate("has", 2, [bill_t], [peach_t])
has2_t = Predicate("has", 2, [joe_t], [pear_t])


#base
john_b1 = Atom("John")     
mary_b1 = Atom("Mary")
apple_b1 = Atom("apple")
loves_b1 = Predicate("loves", 2, [john_b1], [mary_b1])
has_b1 = Predicate("has", 2, [john_b1], [apple_b1])


#Vector encoder using Glove 50d word word embeddings
encoder = Glove50Encoder(r_slots_n, max_ar)    

#encode target
#target_v is a vector represenbtation, target_ps contains the corresponding predicatres
target_v, target_ps = encoder.encode_predicates([loves_t, has1_t, has2_t])
#encode base
base_v, base_p  = encoder.encode_predicates([loves_b1, has_b1])

#create neuual analogical machine
neuro_analogy = NeuralAnalogy(sigma, r_slots_n, max_ar, sem_dim)            
#run mapping
(max_sim, _, mapping) = neuro_analogy.make(target_v, base_v)    


#Print mappings    
for i in range(r_slots_n):
    target_p = target_ps[np.argmax(mapping[:, i])]
    print("{}<->{}".format(target_p, base_p[i]))
    
print("\nSimilarity: {}".format(max_sim))