import numpy as np

from predicates import Atom, Predicate
from encoding import VectorEncoder, Glove50Encoder
from tools import print_v

from  analogy import VectorAnalogy 


            
  
def benchmark():
     
    wm_c = 8
    sem_dim = 50
    max_ar = 2
    
    
    
         
    o1_1 = Atom("o1")
    o2_1 = Atom("o2")

    o1_2 = Atom("o1")
    o2_2 = Atom("o2")

    o1_3 = Atom("o1")
    o2_3 = Atom("o2")

    
    r1_1 = Predicate("r1111", 2, [o1_1], [o2_1])
    r2_1 = Predicate("r2222", 2, [o2_1], [o1_1])
    
    r1_2 = Predicate("r1000", 2, [o2_2], [o1_2])
    r2_2 = Predicate("r2sss", 2, [o1_2], [o2_2])

    r1_3 = Predicate("r1eeee", 2, [o2_3], [o1_3])
    r2_3 = Predicate("r2rrr", 2, [o1_3], [o2_3])

    
    target = Predicate("cause0", 2, [r1_1], [r2_1])
        
    base1 = Predicate("cause1", 2, [r1_2], [r2_2])    
    
    base2 = Predicate("cause2", 2, [r2_3], [r1_3])
        
    #encoder = Glove50Encoder(wm_c, max_ar)    
    encoder = VectorEncoder(wm_c, max_ar, sem_dim)    
    
    base_predicates = [
        [base1],
        [base2], 
    ]
    
    bases = {}
    base_ps = {}
    for i in range(len(base_predicates)):
        vec_repr, ps = encoder.encode_predicates(base_predicates[i])
        bases[i] = vec_repr
        base_ps[i] = ps
        
    target, target_ps = encoder.encode_predicates([target])
    
    #crossed = []
    sigma_i_step = 10
    for sigma_i in range(50, 51, sigma_i_step):        
        #crossed.append(0.0)
        analogy = VectorAnalogy(sigma_i / 100.0, wm_c, max_ar, sem_dim)    
        for base_i in bases:
            (max_sim, _, mapping) = analogy.make(target, bases[base_i])    
            #print_v "None"
#            print([str(p) for p in target_ps])(mapping)
#            print_v(target)
#            print_v(bases[base_i])
#            print(mapping)
#            john_mapping =
            for i in range(len(base_ps[base_i])):
                base_p = base_ps[base_i][i]
                target_p = target_ps[np.argmax(mapping[:, i])]
                print("{}<->{}".format(target_p, base_p))

            
            print("sigma: {:.3}, base: {}, sim :{:.5}".format(sigma_i / 100.0, base_i, max_sim))

#            exit(0)
#        analogy = NeuralAnalogy(sigma_i / 100.0, wm_c, max_ar, sem_dim, False)    
#        t0 = time.time()
#        (max_sim, best_no, mapping) = analogy.make(target, bases)    
#        dt = time.time() - t0
#    
#        mapping_ps = []
#           
#        for i in range(len(base_ps[best_no])):
#            base_p = base_ps[best_no][i]
#            target_p = target_ps[np.argmax(mapping[:,i])]
#            
#            if (base_p <> None or target_p <> None):
#                mapping_ps.append( (target_p, base_p) )
#            if target_p <> None and target_p.same_as(mary):
#                if base_p <> None and base_p.same_as(john):
#                    crossed[int(sigma_i / sigma_i_step)] += 1
#                    
#    print(crossed)                    
#    print(
#        "\n".join(
#                map(lambda(x):"{}<->{}".format(x[0], x[1]), mapping_ps)
#        )
#    )
#    print("")
#    print("{:^5.2f}\t{}\t{:^10.4f}".format(max_sim, best_no, dt))
    
benchmark()
#for wm_c in range(1, 12):
#    NeuralAnalogy(0.5, wm_c, 2, 1, "/cpu:0")
#analogy_o = NeuralAnalogy(0.5, 7, 2, 10, True)
