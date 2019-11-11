from analogy2 import NeuralAnalogy
import numpy as np

from predicates import Atom, Predicate, Property
from encoding import Glove50Encoder, Glove100Encoder, KazumaEncoder, VectorEncoder
from comparison import Comparison

from tools import print_v

#number of representational slots
r_slots_n = 10
#semantics dimensionality (number of units)
sem_dim = 50
#maximal arity
max_ar = 1
#tha relative weight of semantics and structure
sigma = 0.5#semantics and structure have equal weight

#target situation
bill_t = Atom("Bill")
steve_t = Atom("Steve")     
tom_t = Atom("Tom")
smart_bill_t = Property("smart", [bill_t])
tall_bill_t = Property("tall", [bill_t])
smart_steve_t = Property("smart", [steve_t])
timid_tom_t = Property("timid", [tom_t])
tall_tom_t = Property("tall", [tom_t])
same1_t = Predicate("same", 1, [smart_bill_t, smart_steve_t])
same2_t = Predicate("same", 1, [tall_bill_t, tall_tom_t])
unique1_t = Property("unique", [steve_t])
unique2_t = Property("unique", [timid_tom_t])
#base
rover_b1 = Atom("Rover")
fido_b1 = Atom("Fido")     
blackie_b1 = Atom("Blackie")
hungry_rover_b1 = Property("hungry", [rover_b1])
friendly_rover_b1 = Property("friendly", [rover_b1])
hungry_steve_b1 = Property("hungry", [fido_b1])
frisky_blackie_b1 = Property("frisky", [blackie_b1])
friendly_blackie_b1 = Property("friendly", [blackie_b1])
same1_b1 = Predicate("same", 1, [friendly_rover_b1, friendly_blackie_b1])
same2_b1 = Predicate("same", 1, [hungry_rover_b1, hungry_steve_b1])
unique1_b1 = Property("unique", [fido_b1])
unique2_b1 = Property("unique", [frisky_blackie_b1])

#Vector encoder using Glove 50d word word embeddings
#encoder = VectorEncoder(r_slots_n, max_ar, sem_dim)    
encoder = Glove50Encoder(r_slots_n, max_ar)

#encode target
#target_v is a vector represenbtation, target_ps contains the corresponding predicatres
target_v, target_ps = encoder.encode_predicates([smart_bill_t, tall_bill_t, smart_steve_t, timid_tom_t, tall_tom_t, same1_t, same2_t])
#encode base
base_v, base_p  = encoder.encode_predicates([hungry_rover_b1, friendly_rover_b1, hungry_steve_b1,frisky_blackie_b1, friendly_blackie_b1, same1_b1, same2_b1])

#cmp = Comparison(sigma)
#max_sim, target_v_r = cmp.compare(base_v, target_v)
#
#print(max_sim)
#print(target_v_r)
#exit(0)
#create neuual analogical machine
neuro_analogy = NeuralAnalogy(sigma, r_slots_n, max_ar, sem_dim, n_states=362880)            

#run mapping
(max_sim, _, mapping) = neuro_analogy.make(target_v, base_v)    


#Print mappings    
for i in range(r_slots_n):
    target_p = target_ps[np.argmax(mapping[:, i])]
    print("{}<->{}".format(target_p, base_p[i]))
    
print("\nSimilarity: {}".format(max_sim))
