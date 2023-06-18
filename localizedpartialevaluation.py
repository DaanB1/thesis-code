from pgmpy.inference.base import Inference
from interval import interval
from expansion import *
        
class LocalizedPartialEvaluation(Inference):
    
    def __init__(self, model, expansion_method, expansion_rate):
        self.model = model
        self.method = expansion_method
        self.rate = expansion_rate
    
    #The function that starts LPE
    def query(self, node, precision, evidence=None, show_progress=False):
        if show_progress:
            print("Performing LPE")
            print("Query:", node)
            print("Evidence:", evidence)
            print("Precision:", precision)
            print("")
        
        #Prune network and finalize initialization of expansion method
        self.model = self._prune_bayesian_model([node], evidence)[0]
        self.method.set_params(model=self.model, query=node, evidence=evidence)
        self.method.initialize()
        total_nodes = len(self.model.nodes)
        
        #Initialize active set
        for n in self.model.nodes:
            self.model.nodes[n]['active'] = False
        self.model.nodes[node]['active'] = True
        set_size = 1
        if isinstance(self.rate, AdaptiveRate):
            self.rate.set_params(set_size, precision)
            result = [interval[0, 1]] * self.model.get_cardinality(node)
            self.rate.add_result(result)
        set_size += self.method.expand(self.rate.get_rate(model))
        
        #Propagate to find interval. Expand active set. Repeat until precision is met.      
        intsize = float('inf')
        iteration = 0
        while True:
            result = self.propagate(node, evidence)
            if isinstance(self.rate, AdaptiveRate):
                self.rate.add_result(result)
            if show_progress:
                print("Iteration", str(iteration) + ".", "Considering", set_size, "/", total_nodes, "nodes.")
                for i, state in enumerate(self.model.states[node]):
                    print(state + ":", result[i])
                print("")
                
            intsize = 0
            for interval_ in result:
                s = interval_[0].sup - interval_[0].inf
                if s > intsize:
                    intsize = s
            if intsize <= precision:
                return result
            
            set_size += self.method.expand(self.rate.get_rate(model))
            iteration += 1
    
    #Belief propagation using intervals and an active set
    def propagate(self, query, evidence=None, normalize=True):
        
        #Calculates the lambda values of node
        def la(node):
            if evidence != None and node in evidence:
                child_messages = [list(send_la_message(child, node).values()) for child in self.model.get_children(node)]
                la = [interval(0)] * self.model.get_cardinality(node)
                la = makeDict(self.model.states[node], la)
                la[evidence[node]] = interval(1)
                
                if len(child_messages) == 0:
                    return la
                else:
                    child_messages.append(list(la.values()))
                    la = prod(child_messages)
                    return makeDict(self.model.states[node], normalized(la))
                                    
            child_messages = [list(send_la_message(child, node).values()) for child in self.model.get_children(node)]
            if len(child_messages) == 0:
                la = [interval(1)] * self.model.get_cardinality(node)
            else:
                la = prod(child_messages)
                
            return makeDict(self.model.states[node],  normalized(la))#normalized(la))

        #Calculates the pi value of node
        def pi(node):
            if evidence != None and node in evidence:
                pi = np.zeros(shape=self.model.get_cardinality(node))
                pi = makeDict(self.model.states[node], pi)
                pi[evidence[node]] = interval(1)
                return pi
            
            pi_messages = [send_pi_message(parent, node) for parent in self.model.get_parents(node)]
            pi_messages = makeDict(self.model.get_parents(node), pi_messages)
            cpt = self.model.get_cpds(node)
            values = cpt.get_values().flatten()
            assignments = cpt.assignment(range(len(values)))
            results = makeDict(self.model.states[node], [[] for i in range(len(self.model.states[node]))])

            for index, value in enumerate(values):
                assignment = dict(assignments[index])
                message_product = interval(1)
                for parent in self.model.get_parents(node):
                    message_product *= pi_messages[parent][assignment[parent]]
                results[assignment[node]].append((interval(value), interval(message_product)))

            pi = makeDict(self.model.states[node], [None] * len(self.model.states[node]))
            for state, pairs in results.items():
                pi[state] = annihilationReinforcement(pairs)
                
            return pi

        #node1 (child) sends a lambda message to node2 (parent)
        def send_la_message(node1, node2):
            
            if not self.model.nodes[node1]['active']:
                la_message = [interval[0, 1]] * self.model.get_cardinality(node1)
                return makeDict(self.model.states[node1], la_message)
            
            other_parents = self.model.get_parents(node1)
            other_parents.remove(node2)
            pi_messages = [send_pi_message(parent, node1) for parent in other_parents]
            pi_messages = makeDict(other_parents, pi_messages)
            l = la(node1)
            
            cpt = self.model.get_cpds(node1)
            values = cpt.get_values().flatten()
            assignments = cpt.assignment(range(len(values)))
            s1 = self.model.states[node1]
            s2 = self.model.states[node2]
            results = makeDict(s2, [makeDict(s1, [[] for i in s1]) for i in s2])
            
            for index, value in enumerate(values):
                assignment = dict(assignments[index])
                message_product = interval(1)
                for parent in other_parents:
                    message_product *= pi_messages[parent][assignment[parent]]
                results[assignment[node2]][assignment[node1]].append((interval(value), interval(message_product)))
                
            la_message = makeDict(self.model.states[node2], [None] * len(self.model.states[node2]))
            for state2, value in results.items():
                new_pairs = []
                for state1, pairs in value.items():
                    new_pairs.append((annihilationReinforcement(pairs), l[state1]))
                la_message[state2] = annihilationReinforcement(new_pairs)
                    
            return la_message

        #node1 (parent) sends a pi message to node2 (child)
        def send_pi_message(node1, node2):
            if not self.model.nodes[node1]['active']:
                pi_message = [interval[0, 1]] * self.model.get_cardinality(node1)
                return makeDict(self.model.states[node1], pi_message)
            if evidence != None and node1 in evidence:
                return pi(node1)
                
            other_children = self.model.get_children(node1)
            other_children.remove(node2)
            la_messages = [list(send_la_message(child, node1).values()) for child in other_children]
            la_messages.append(list(pi(node1).values()))
            pi_message = prod(la_messages)
            return makeDict(self.model.states[node1], normalized(pi_message))#normalized(pi_message))
        
        #---HELPER FUNCTIONS---
        
        #takes a list of keys and a list of values and builds a dictionary
        def makeDict(states, values):
            if len(states) != len(values):
                raise Exception("ERROR", states, values)
            return dict(map(lambda i,j : (i,j) , states,values))
        
        #Multiplies all the values in each column together
        def prod(matrix):
            if len(matrix) == 0:
                return []
            result = [1] * len(matrix[0])
            for i in range(len(matrix[0])):
                for j in range(len(matrix)):
                    result[i] = matrix[j][i] * result[i]
            return result
        
        #Normalizes a vector of intervals
        def normalized(message):
            copy = message.copy()
            removed = []
            total_l = 0
            total_u = 0
            result = []
            for i, inter in reversed(list(enumerate(message))):
                total_l += inter[0].inf
                total_u += inter[0].sup
                if inter[0].inf == 0 and inter[0].sup == 0:
                    del copy[i]
                    removed.append(i)    
            for i in copy:
                u = i[0].sup / (i[0].sup - i[0].inf + total_l)
                if i[0].inf - i[0].sup + total_u == 0:
                    l = u
                else:
                    l = i[0].inf / (i[0].inf - i[0].sup + total_u)
                result.append(interval[l, u])
            for i in reversed(removed):
                result.insert(i, interval[0])
            return result                      
        
        #Applies the A/R algorithm to vectors a and b zipped together in the list pairs
        #Pairs is a list of (probability, message)
        def annihilationReinforcement(pairs):
            lowerbound_sum = sum([pair[1][0].inf for pair in pairs])
            upperbound_sum = sum([pair[1][0].sup for pair in pairs])

            #calculate lowerbound
            pairs.sort(key = lambda x: x[0][0].inf)
            l = 0
            mass = lowerbound_sum
            for pair in pairs:
                increase = pair[1][0].sup - pair[1][0].inf
                if increase + mass > 1:
                    increase = 1 - mass
                mass += increase
                l += pair[0][0].inf * (pair[1][0].inf + increase)

            #calculate upperbound
            pairs.sort(key = lambda x: x[0][0].sup)
            u = 0
            mass = upperbound_sum
            for pair in pairs:
                decrease = pair[1][0].sup - pair[1][0].inf
                if mass - decrease < 1:
                    decrease = mass - 1
                mass -= decrease
                u += pair[0][0].sup * (pair[1][0].sup - decrease)

            return interval[l, u]
        
        #Start the propagation process and return BEL(x)
        l = list(la(query).values())
        p = list(pi(query).values()) 
        return normalized(prod([l, p]))