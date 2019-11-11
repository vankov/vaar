from analogy import NeuralAnalogy


import numpy as np



from predicates import Atom, Predicate
from encoding import VectorEncoder, Glove50Encoder

import time

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-rn', type=int, help='Number of r slots', default=7)                    
parser.add_argument('-reps', type=int, help='number of repetions', default=1)
parser.add_argument('-min_n', type=int, help='minimal size of knowledge base', default=1)
parser.add_argument('-max_n', type=int, help='maximal size of knowledge base', default=1)
parser.add_argument('-step', type=int, help='step', default=1)
parser.add_argument('-s', type=float, help='sigma', default=0.5)
parser.add_argument('-csv', help='output data in csv format', action='store_true', default=False)
args = parser.parse_args()

#from tools import print_v



     
wm_c = args.rn
sem_dim = 50
max_ar = 2



#target situation
person_t = Atom("person")     
dog_t = Atom("dog")
barks_at_t = Predicate("barks", 2, [dog_t], [person_t])
scares_t = Predicate("scares", 2, [dog_t], [person_t])
causes_t = Predicate("causes", 2, [barks_at_t], [scares_t])

#base 1
person_1_b1 = Atom("person")     
person_2_b1 = Atom("person")     
dog_1_b1 = Atom("dog")
dog_2_b1 = Atom("dog")
barks_at_b1 = Predicate("barks", 2, [dog_1_b1], [person_1_b1])
scares_b1 = Predicate("scares", 2, [dog_2_b1], [person_2_b1])
causes_b1 = Predicate("causes", 2, [barks_at_b1], [scares_b1])


#base 2
person_b2 = Atom("person")     
dog_b2 = Atom("dog")
barks_at_b2 = Predicate("barks", 2, [dog_b2], [person_b2])
scares_b2 = Predicate("scares", 2, [person_b2], [dog_b2])
causes_b2 = Predicate("causes", 2, [scares_b2], [barks_at_b2])

#base 3
person_b3 = Atom("person")     
dog_b3 = Atom("dog")
loves_1_b3 = Predicate("loves", 2, [dog_b3], [person_b3])
loves_2_b3 = Predicate("loves", 2, [person_b3], [dog_b3])
causes_b3 = Predicate("causes", 2, [loves_1_b3], [loves_2_b3])

#base 4
person_b4 = Atom("person")     
dog_b4 = Atom("dog")
barks_at_b4 = Predicate("barks", 2, [dog_b4], [person_b4])
scares_b4 = Predicate("scares", 2, [dog_b4], [person_b4])
  
#base 5
person_b5 = Atom("person")     
dog_b5 = Atom("dog")
barks_at_b5 = Predicate("barks", 2, [dog_b5], [person_b5])
scares_b5 = Predicate("scares", 2, [dog_b5], [person_b5])
causes_1_b5 = Predicate("causes", 2, [barks_at_b5], [scares_b5])
hates_b5 = Predicate("hates",2, [person_b5], [dog_b5])
causes_2_b5 = Predicate("causes", 2, [causes_1_b5], [hates_b5])


encoder = Glove50Encoder(wm_c, max_ar)    
#encoder = VectorEncoder(wm_c, max_ar, sem_dim)    

target_v, target_ps = encoder.encode_predicates([causes_t])

base_predicates = [
    [causes_b1], 
    [causes_b2], 
    [causes_b3], 
    [barks_at_b4, scares_b4], 
    [causes_2_b5]
]

bases_v = []
bases = {}
for i in range(len(base_predicates)):
    bases[i] = encoder.encode_predicates(base_predicates[i])
    bases_v.append(bases[i][0])
    
sigma_i_step = 5

def benchmark_knowledge_base_size(target, bases, min_n = 1, max_n = 1, step = 1, reps = 1):
    sigma = 0.5
    neuro_analogy = NeuralAnalogy(sigma, wm_c, max_ar, sem_dim)    
    neuro_analogy.make(target, bases)
    
    results = []
    for i in range(min_n, max_n + 1, step):
        KB = []
        
        while len(KB) < i:
            b_rnd = np.random.randint(0, len(bases))
            KB.append(bases[b_rnd])
            
        if len(KB) > 0:
            dt = 0
            for r in range(reps):
                time1 = time.time()
                neuro_analogy.make(target, KB)    
                dt += time.time() - time1
            results.append((len(KB), dt / float(reps)))
            
            
    return results

results = benchmark_knowledge_base_size(
        target = target_v, 
        bases = bases_v, 
        min_n = args.min_n, 
        max_n = args.max_n, 
        step = args.step, 
        reps = args.reps)

args = parser.parse_args()

if not(args.csv):
    print("{:>5}\t{}".format("N", "Time (sec)"))
    
for i in range(len(results)):
    if not(args.csv):
        print("{:>5}\t{:.4}".format(results[i][0], results[i][1]))
    else:
        print("{},{:.8}".format(results[i][0], results[i][1]))
        
exit(0)        

print("---------------Analogy-----------")
print("\n")

for sigma_i in range(50, 51, sigma_i_step):        
    neuro_analogy = NeuralAnalogy(sigma_i / 100.0, wm_c, max_ar, sem_dim)    
#    print("---------------Mapping-----------")    
#    for base_i in bases:
#        (max_sim, _, mapping) = analogy.make(target_v, bases[base_i][0])    
#        for i in range(len(bases[base_i][1])):
#            base_p = bases[base_i][1][i]
#            target_p = target_ps[np.argmax(mapping[:, i])]
#            print("{}<->{}".format(target_p, base_p))
#
#        
#        print("sigma: {:.3}, base: {}, sim :{:.5}".format(sigma_i / 100.0, base_i, max_sim))
#        print("\n")

    time1 = time.time()
    (max_sim, best_base_no, mapping) = neuro_analogy.make(target_v, bases_v)    
    time2 = time.time()
    person_mapping = "None"
    dog_mapping = "None"
    for i in range(len(bases[best_base_no][1])):
        base_p = bases[best_base_no][1][i]
        target_p = target_ps[np.argmax(mapping[:, i])]
        if target_p <> None:
            if str(target_p) == "person-0":
                person_mapping = base_p
            if str(target_p) == "dog-0":
                dog_mapping = base_p
        #print("{}<->{}".format(target_p, base_p))
    
    print("Sigma: {:.4f}\tBest Base: {}\tSim: {:.4}\tTime:{:.4}s\tPerson<->{}\tDog<->{}".format(
            sigma_i / 100.0, 
            best_base_no, 
            max_sim,
            (time2 - time1),
            person_mapping,
            dog_mapping
            ))
#    for i in range(len(bases[best_base_no][1])):
#        base_p = bases[best_base_no][1][i]
#        target_p = target_ps[np.argmax(mapping[:, i])]
#        print("{}<->{}".format(target_p, base_p))
#    print("\n")    
    