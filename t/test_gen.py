import tensorflow as tf
import numpy as np
import scipy.spatial.distance
from scipy.special import expit

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import Callback

from tools import print_v


from predicates import Atom, Predicate

class ProgressTracker(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
                    
    def report(self, (acc, cos_sim), (v_acc, v_cos_sim)):
        loss = self.losses[len(self.losses) - 1]
                
        print("{:>7}{:>7.2f}%\tLoss: {:.5f}\tACC:{:.5f}\tCos:{:.5f}\tvACC:{:.5f}\tvCos:{:.5f}".format(
                self.epoch, 
                (self.epoch / float(self.max_epoch)) * 100,
                loss,  
                acc,
                cos_sim,
                v_acc,
                v_cos_sim))
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def __init__(self, max_epoch):
        Callback.__init__(self)
        self.epoch = 0
        self.max_epoch = max_epoch
        
class GenNet:
    
    def gen_batch(self, data, batch_size):
        
        sel = np.random.choice(len(data[0]), size=batch_size, replace=False)
        return (
            np.array([data[0][x] for x in sel]),
            np.array([data[1][x] for x in sel])
        )
        
    def save(self, filename = "model.state.h5"):
        self.__model.save(filename)
        
    def load_weights(self, filename = "model.state.h5"):
        self.__model.load_weights(filename);
        
    def calc_accuracy(self, inputs, targets, print_error = False):
        outputs = self.__model.predict(inputs, batch_size=len(inputs))
             
        sum_s = 0
        sum_cosine_s = 0
            
        for i in range(len(inputs)):
            sem_target = \
                targets[i,:self.__r_slots_n * self.__sem_dim].reshape(
                        self.__r_slots_n, 
                        self.__sem_dim)
            struct_target = \
                targets[i,self.__r_slots_n * self.__sem_dim:].reshape(
                        self.__max_arity, 
                        self.__r_slots_n, 
                        self.__r_slots_n)
                
            sem_output = outputs[i,:self.__r_slots_n * self.__sem_dim].reshape(
                        self.__r_slots_n, 
                        self.__sem_dim)
            struct_output = \
                outputs[i,self.__r_slots_n * self.__sem_dim:].reshape(
                        self.__max_arity, 
                        self.__r_slots_n, 
                        self.__r_slots_n)
            
            
            sem_output = expit(sem_output)
            struct_output = expit(struct_output)
            
            sem_matches_all = 0
            struct_matches_all = 0
            sem_matches = 0
            struct_matches = 0
             

            for j in range(self.__r_slots_n):
                max_indices = sem_output[j].argsort()[::-1][0:int(np.sum(sem_target[j]))]

                    
                for k in max_indices:
                    sem_matches_all += 1
                    if sem_target[j][k]:
                        sem_matches += 1                            
                if np.sum(struct_target[0, j]):
                    max_indices = struct_output[0, j].argsort()[::-1][0:int(np.sum(struct_target[0, j]))]
                    for k in max_indices:
                        struct_matches_all += 1
                        if struct_target[0, j, k]:
                            struct_matches += 1         
                if np.sum(struct_target[1, j]):
                    max_indices =struct_output[1, j].argsort()[::-1][0:int(np.sum(struct_target[1, j]))]
                    for k in max_indices:
                        struct_matches_all += 1
                        if struct_target[1, j, k]:
                            struct_matches += 1                                                            
             
            if (sem_matches == sem_matches_all) and (struct_matches == struct_matches_all):
                s = 1.0
            else:
                s = 0.0
                            
            cos_s = self.__sigma * (1 - scipy.spatial.distance.cosine(
                    sem_output.flatten(), 
                    sem_target.flatten()))
            
            if (np.sum(struct_target.flatten())):
                cos_s += self.__sigma * (1 - scipy.spatial.distance.cosine(
                        struct_output.flatten(), 
                        struct_target.flatten()))                
            else:
                cos_s += 0.5

            if (print_error) and (s == 0):
                print(sem_matches, sem_matches_all)
                print(struct_matches, struct_matches_all)
                print(cos_s)                
                
                print_v((sem_target, struct_target))
                print_v((sem_output, struct_output), precision = 1)

                exit(0)

                
            sum_s += s
            sum_cosine_s += cos_s
            
        return (sum_s / len(inputs), sum_cosine_s / len(inputs))
        
    def train(self, train_data, valid_data, epoch_init = 0, epochs_n = 1000, steps_per_epoch = 10, batch_size=1):
        
        progress_tracker = ProgressTracker(epochs_n)
        max_v_acc = 0
                
        callbacks=[progress_tracker]
        if (epoch_init > 0):
            epoch_init -= 1
        for e in range(epoch_init, epochs_n):            
            progress_tracker.set_epoch(e + 1)
            batches = 0
            
            while batches < steps_per_epoch:
                (inputs, targets) = self.gen_batch(train_data, batch_size)
                

                self.__model.fit(
                        x = np.array(inputs), 
                        y = np.array(targets),
                        batch_size=batch_size,
                        verbose=0,
                        callbacks=callbacks
                        );                
                batches += 1
            
            (acc, cos) = self.calc_accuracy(inputs, targets)
            (v_inputs, v_targets) = self.gen_batch(valid_data, batch_size)
            (v_acc, v_cos) = self.calc_accuracy(v_inputs, v_targets)
            
#            if (acc > 0.99):
#                self.calc_accuracy(v_inputs, v_targets, print_error = True)
                
            progress_tracker.report((acc, cos), (v_acc, v_cos))
            self.save()
            if v_acc > max_v_acc:
                max_v_acc = v_acc
                self.save(filename="best.model.state.h5")
                
    def __init__(self, input_dim, r_slots_n, sem_dim, max_arity):
            self.__input_dim = input_dim
            self.__r_slots_n = r_slots_n
            self.__max_arity = max_arity
            self.__sem_dim = sem_dim
            self.__sigma = 0.5
            
            def loss_f(y_true, y_pred):            
                loss = \
                    (1 - self.__sigma) * tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=y_pred[:,:self.__r_slots_n * self.__sem_dim], 
                                labels=y_true[:,:self.__r_slots_n * self.__sem_dim])
                    ) + \
                    self.__sigma * tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=y_pred[:,self.__r_slots_n * self.__sem_dim:], 
                                labels=y_true[:,self.__r_slots_n * self.__sem_dim:])
                    )
                return loss
            
            self.__model=Sequential()
            self.__model.add(
                Dense(
                        units=100, 
                        activation  =  "tanh", 
                        input_dim = self.__input_dim))
            self.__model.add(
                Dense(
                        units = self.__r_slots_n * (self.__sem_dim + self.__max_arity * self.__r_slots_n)
                        ))
            
            self.__model.compile(loss = loss_f, optimizer = Adam(lr=0.0001))
            
class Settings:
    def __init__(self):
        self.r_slots_n = 4
        
class Stimuli:

    def get_trainset_size(self):
        return len(self.__trainset_inputs_p)
    
    def get_train_set(self):
        return (self.__trainset_inputs_p, np.array(self.__trainset_inputs_v)),  (self.__trainset_targets_p, np.array(self.__trainset_targets_v))
    
    def get_test_set(self):
        return (self.__testset_inputs_p, np.array(self.__testset_inputs_v)),  (self.__testset_targets_p, np.array(self.__testset_targets_v))
    
    def __init__(self, r_slots_n, max_arity, sem_dim):
        vocab = {
            "John": 0,
            "Mary": 1,
            "Bill": 2,
            "Emma": 3,
            "Peter": 4,
            "Maria": 5,
            "Andrea": 6,
            "Tommy": 7,
            "George": 8,
            "Susan": 9,
            
            "give": 10, #give(S, O , P) -> have(P, O),
            "steal_from": 11, #steal_from(S,O, P) -> angry_at(P, S)
            "ask_for": 12, #ask_for(S, O, P) -> return(P, O, S)
            
            "book": 13,
            "pen": 14,
            "ring": 15,
            "cup": 16,
            "apple": 17,            
            "camera": 18,
            "bottle": 19,
            "purse": 20,
            "bag": 21,
            "chair": 22,
            
            "has": 23,
            "return": 24,
            "angry_at": 25,
            
            "John2": 26,
            "Mary2": 27,
            "Bill2": 28,
            "Emma2": 29,            
            "Peter2": 30,
            "Maria2": 31,
            "Andrea2": 32,
            "Tommy2": 33,
            "George2": 34,
            "Susan2": 35
        }

                
        self.__testset_inputs_v = []
        self.__testset_targets_v = []        
        self.__trainset_inputs_v = []
        self.__trainset_targets_v = []
        
        for j in range(1000):
            for subject_str in ["John", "Mary", "Bill", "Emma", "Peter", "Maria", "Andrea", "Tommy", "George", "Susan"]:
                for action_str in ["give", "steal_from", "ask_for"]:
                    for patient_str in ["John", "Mary", "Bill", "Emma", "Peter", "Maria", "Andrea", "Tommy", "George", "Susan"]:
                        if subject_str == patient_str:
                            continue
                        for object_str in ["book", "pen", "ring", "cup", "apple", "camera", "bottle", "purse", "bag", "chair"]:
                            
                            input_v = np.zeros(shape=36 + 4)                        
                            input_v[vocab[subject_str] = 1
                            input_v[vocab[action_str]] = 1
                            input_v[vocab[object_str]] = 1
                            input_v[vocab["{}2".format(patient_str)]] = 1
                            
                            target_v_sem = np.zeros(shape=(r_slots_n, 26))
                            target_v_struct = np.zeros(shape=(max_arity, r_slots_n, r_slots_n))
                            
                            rnd = np.random.choice(r_slots_n, size=4, replace = False)
    
                            input_v[36:] = rnd
                            
                            if action == "ask_for":
                                
                                target_v_sem[rnd[0],vocab["return"]] = 1                
                                target_v_sem[rnd[1],vocab[subject.get_type_id()]] = 1
                                target_v_sem[rnd[2],vocab[object_p.get_type_id()]] = 1
                                target_v_sem[rnd[3],vocab[patient.get_type_id()]] = 1
                                
                                target_v_struct[0, rnd[0], rnd[3]] = 1                
                                target_v_struct[1, rnd[0], rnd[2]] = 1                
                                target_v_struct[2, rnd[0], rnd[1]] = 1                
                                
                            if action == "give":
                                
                                target_v_sem[rnd[0],vocab["has"]] = 1                        
                                target_v_sem[rnd[2],vocab[object_p.get_type_id()]] = 1
                                target_v_sem[rnd[3],vocab[patient.get_type_id()]] = 1                            
                                
                                target_v_struct[0, rnd[0], rnd[3]] = 1                
                                target_v_struct[1, rnd[0], rnd[2]] = 1                                    
                                
                            if action == "steal_from":
                                target_p = Predicate("angry_at", 2, [patient], [subject])
                                target_v_sem[rnd[0],vocab["angry_at"]] = 1           
                                target_v_sem[rnd[1],vocab[subject.get_type_id()]] = 1
                                target_v_sem[rnd[3],vocab[patient.get_type_id()]] = 1                            
                                                            
                                target_v_struct[0, rnd[0], rnd[3]] = 1                
                                target_v_struct[1, rnd[0], rnd[1]] = 1                                    
    
    #                        print(target_p)
    #                        print_v((target_v_sem, target_v_struct))
    #                        print(input_v)
                            
                            target_v = np.concatenate((target_v_sem.flatten(), target_v_struct.flatten()))
                            
                            if (action == "give" and patient.same_as(mary)):
                                self.__testset_inputs_p.append(input_p)
                                self.__testset_targets_p.append(target_p)
                                self.__testset_inputs_v.append(input_v)
                                self.__testset_targets_v.append(target_v)
                            else:
                                self.__trainset_inputs_p.append(input_p)
                                self.__trainset_targets_p.append(target_p)
                                self.__trainset_inputs_v.append(input_v)
                                self.__trainset_targets_v.append(target_v)
                                
stimuli = Stimuli(4, 3, 14)
trainset_inputs, trainset_targets = stimuli.get_train_set()
testset_inputs, testset_targets = stimuli.get_test_set()

net = GenNet(input_dim=22, r_slots_n = 4, sem_dim = 14, max_arity = 3)
net.train(
        train_data=(trainset_inputs[1], trainset_targets[1]),
        valid_data=(testset_inputs[1], testset_targets[1]),
        steps_per_epoch=stimuli.get_trainset_size() / 200,
        batch_size=200)

    #def train(self, train_data, valid_data, epoch_init = 0, steps_per_epoch = 10, batch_size=1):
    
#for i in range(len(testset_inputs[0])):
#    print("{} --> {}".format(testset_inputs[0][i], testset_targets[0][i]))
#    print(testset_inputs[1][i])
#    print_v(trainset_targets[1][i])

