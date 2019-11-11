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
john_t = Atom("John")     
mary_t = Atom("Mary")
loves_t = Predicate("loves", 2, [john_t], [mary_t])

#base 1
john_b1 = Atom("John")     
maria_b1 = Atom("Maria")
loves_b1 = Predicate("loves", 2, [john_b1], [maria_b1])

#base 2
peter_b2 = Atom("Peter")     
maria_b2 = Atom("Maria")
loves_b2 = Predicate("loves", 2, [peter_b2], [maria_b2])


#base 3
john_b3 = Atom("John")     
mary_b3 = Atom("Mary")
hates_b3 = Predicate("hates", 2, [john_b3], [mary_b3])


#base 4
john_b4 = Atom("John")     
woman_b4 = Atom("woman")
loves_b4 = Predicate("loves", 2, [john_b4], [woman_b4])

#base 5
john_b5 = Atom("John")     
cooking_b5 = Atom("cooking")
loves_b5 = Predicate("loves", 2, [john_b5], [cooking_b5])

#base 6
john_b6 = Atom("John")     
mary_b6 = Atom("Mary")
serves_b6 = Predicate("serves", 2, [john_b6], [mary_b6])

#base 7
john_b7 = Atom("John")     
mary_b7 = Atom("Mary")
invents_b7 = Predicate("invents", 2, [john_b7], [mary_b7])

#base 8
boy_1_b8 = Atom("boy")     
boy_2_b8 = Atom("boy")
plays_b8 = Predicate("plays", 2, [boy_1_b8], [boy_2_b8])

#base 9
chimney_b9 = Atom("chimney")     
house_b9 = Atom("house")
on_b9 = Predicate("on", 2, [chimney_b9], [house_b9])

#base 10
john_b10 = Atom("John")     
mary_b10 = Atom("Mary")
loves_b10 = Predicate("loves", 2, [mary_b10], [john_b10])


encoder = Glove50Encoder(wm_c, max_ar)    

target_v, target_ps = encoder.encode_predicates([loves_t])

base_labels = [
        "John loves Maria.",
        "Peter loves Maria.",
        "John hates Maria.",  
        "John loves a woman",
        "John loves cooking.",                
        "John serves Maria.",  
        "John invents Maria.",
        "A boy plays with another boy.",
        "A chimney is on a house.",                                        
        "Mary loves John.",
]

base_predicates = [
    [loves_b1], 
    [loves_b2], 
    [hates_b3], 
    [loves_b4], 
    [loves_b5],
    [serves_b6], 
    [invents_b7], 
    [plays_b8],    
    [on_b9], 
    [loves_b10],    
]

bases_v = []
bases = {}
for i in range(len(base_predicates)):
    bases[i] = encoder.encode_predicates(base_predicates[i])
    bases_v.append(bases[i][0])
    

traces = {
        "john": {"corr": [], "cross": [], "rel": [], "none": []},
        "mary": {"corr": [], "cross": [], "rel": [], "none": []} 
        }

base_bars = {}


for base_i in bases:
    for t in traces["john"]:
        traces["john"][t].append((None, None))
        traces["mary"][t].append((None, None))

    base_bars[base_i] = {"john": {"corr": [], "cross": [], "rel": [], "none": []}}
    
    for sigma_i in range(0, 101, 10):        
        
        neuro_analogy = NeuralAnalogy(sigma_i / 100.0, wm_c, max_ar, sem_dim)            

        
            
        (max_sim, _, mapping) = neuro_analogy.make(target_v, bases[base_i][0])    
                
        
        t = [ 
                (bases[base_i][1][i].get_arguments()[0], bases[base_i][1][i].get_arguments()[1]) 
                for i in range(len(bases[base_i][1])) 
                if (type(bases[base_i][1][i]) is Predicate) and (bases[base_i][1][i].get_arity() == 2)
                ][0]
        args = [t[0][0],t[1][0]]
        
        for i in range(len(bases[base_i][1])):
            base_p = bases[base_i][1][i]
            target_p = target_ps[np.argmax(mapping[:, i])]
            #print("{}<->{}".format(target_p, base_p))
            if target_p <> None:
                if str(target_p.get_id()) == "John-0":
                    if base_p <> None and (base_p.same_as(args[0])):
                        traces["john"]["corr"].append((sigma_i / 100.0, max_sim))
                        base_bars[base_i]["john"]["corr"].append((sigma_i / 100.0, max_sim))
                    else:
                        if base_p <> None and base_p.same_as(args[1]):
                            traces["john"]["cross"].append((sigma_i / 100.0, base_i * 2 + max_sim))
                            base_bars[base_i]["john"]["cross"].append((sigma_i / 100.0, max_sim))
                        else:
                            if base_p <> None:
                                traces["john"]["rel"].append((sigma_i / 100.0, base_i * 2 + max_sim))
                                base_bars[base_i]["john"]["rel"].append((sigma_i / 100.0, max_sim))
                            else:
                                traces["john"]["none"].append((sigma_i / 100.0, base_i * 2 + max_sim))
                                base_bars[base_i]["john"]["none"].append((sigma_i / 100.0, max_sim))
                                
                if str(target_p.get_id()) == "Mary-0":
                    if base_p <> None and (base_p.same_as(args[1])):
                        traces["mary"]["corr"].append((sigma_i / 100.0, base_i * 2 + max_sim))
                    else:
                        if base_p <> None and base_p.same_as(args[1]):
                            traces["mary"]["corr"].append((sigma_i / 100.0, base_i * 2 + max_sim))
                        else:
                            if base_p <> None:
                                traces["mary"]["corr"].append((sigma_i / 100.0, base_i * 2 + max_sim))
                            else:
                                traces["mary"]["corr"].append((sigma_i / 100.0, base_i * 2 + max_sim))
#                if str(target_p.get_id()) == "loves-0":
#                    if base_p <> None and (base_p.same_as(args[0]) or base_p.same_as(args[1])):
#                        mappings[sigma_i][base_i]["loves"]["object"] += 1
#                    else:
#                        if base_p <> None:
#                            mappings[sigma_i][base_i]["loves"]["corr"] += 1
#                        else:
#                            mappings[sigma_i][base_i]["loves"]["none"] += 1
                        
                
#        print("sigma: {:.3}, base: {}, sim :{:.5}".format(sigma_i / 100.0, base_i, max_sim))
        #print("\n")
        #exit(0)

colors = {
        "corr": "rgb(0,100,0)", 
        "cross" : "rgb(200,0,0)", 
        "rel": "rgb(0, 0, 200)", 
        "none": "rgb(50,50,50)"}
bars = []
annotations = []

for base_i in base_bars:
    #print(base_bars[base_i]["john"])
    filler = [[], []]
    
    for m in base_bars[base_i]["john"]:
        xs = []
        ys = []        
        for xy_i in range(len(base_bars[base_i]["john"][m])):
            xs.append(base_bars[base_i]["john"][m][xy_i][0])    
            ys.append(base_bars[base_i]["john"][m][xy_i][1])    
            filler[0].append(base_bars[base_i]["john"][m][xy_i][0])
            filler[1].append(1 - base_bars[base_i]["john"][m][xy_i][1] + 0.5)
            
        print("trace_{0}_{3} = Bar(x=[{1}], y=[{2}], name='{0}', marker=dict(color='{4}'))".format(
                m, 
                ", ".join(map(str, xs)),
                ", ".join(map(str, ys)),
                base_i,
                colors[m]
                ))
        bars.append("trace_{}_{}".format(m, base_i))    
        
    print("filler_trace_{0} = Bar(x=[{1}], y=[{2}], marker=dict(color='rgba(1,1,1,0)'))".format(
            base_i,
            ", ".join(map(str, filler[0])),
            ", ".join(map(str, filler[1]))
            ))
        
    bars.append("filler_trace_{}".format(base_i))
    annotations.append(
            (int(base_i) * 1.5 + 0.5, base_labels[base_i])
            )
#    annotations.append(dict(xref='paper', x=0, y=0.5,
#                                  xanchor='right', yanchor='middle',
#                                  text='Mary loves John',
#    
#print                              showarrow=False))
print("\nannotations = []")
for a_i in range(len(annotations)):
    print("annotations.append(dict(xref='paper', x=0, y={}, xanchor='right', yanchor='middle', text='{}', showarrow=False))". format(annotations[a_i][0], annotations[a_i][1]))
    
print("\ndata = [{}]".format(", ".join(bars)))        
exit(0)            
for tr in traces["john"]:
    xs = []
    ys = []
    for xy_i in range(len(traces["john"][tr])):
        xs.append(traces["john"][tr][xy_i][0])    
        ys.append(traces["john"][tr][xy_i][1])    
        
    print("\ntrace_{0} = Scatter(x=[{1}], y=[{2}], name='{0}', connectgaps=False)\n\n".format(
            tr, 
            ", ".join(map(str, xs)),
            ", ".join(map(str, ys)),
            ))
print(traces["john"])        