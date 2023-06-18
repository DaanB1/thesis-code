from expansion import *
from localizedpartialevaluation import *
from pgmpy.readwrite import XMLBIF
import random

#Generates a random query and set of evidence nodes
def generate_input(model):
    evidence = {}
    amount = random.randint(0, len(model.nodes) // 4)
    nodes = list(model.nodes)
    query = nodes[random.randint(0, len(model.nodes) - 1)]
    for i in range(amount):
        node = nodes[random.randint(0, len(nodes) - 1)]
        if node == query:
            continue
        states = model.states[node]
        state = states[random.randint(0, len(states) - 1)]
        evidence[node] = state
    return query, evidence


#Load network from file
model = XMLBIF.XMLBIFReader("model.xml").get_model()

#Expansion methods
expansion_method = BreadthFirstExpansion()
expansion_method = TargetedExpansion(target="roots")
expansion_method = TargetedExpansion(target="evidence")
expansion_method = TargetedExpansion(target="both")

#Expansion rates
constant = 1 / 10
rate_method = ConstantRate(constant)
rate_method = AdaptiveRate(constant)

#Perform LPE
precision = 0.1
query, evidence = generate_input(model)
lpe = LocalizedPartialEvaluation(model, expansion_method, rate_method)
result = lpe.query(query, precision, evidence, show_progress=False)
print(result)
