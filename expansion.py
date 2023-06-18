class Expansion():
    
    def __init__(self):
        self.model = None
        self.evidence = None
        self.query = None
        
    def set_params(self, model=None, evidence=None, query=None):
        self.model = model
        self.evidence = evidence
        self.node=query
    
class BreadthFirstExpansion(Expansion):
    
    def __init__(self):
        self.queue = None
        self.visited = None
        
    def initialize(self):
        self.queue = [self.node]
        self.visited = dict(map(lambda i,j : (i,j) , self.model.nodes, [False] * len(self.model.nodes)))
    
    def expand(self, rate):
        if len(self.queue) == 0:
            raise Exception("Error: empty queue")
        added_nodes = 0
        while not len(self.queue) == 0:
            node = self.queue.pop(0)
            self.visited[node] = True

            for neighbour in self.model.get_children(node) + self.model.get_parents(node):
                if not self.visited[neighbour]:
                    self.queue.append(neighbour)
            if not self.model.nodes[node]['active']:
                self.model.nodes[node]['active'] = True
                added_nodes += 1
                if added_nodes == rate:
                    return added_nodes
        return  added_nodes
        
class TargetedExpansion(Expansion):
    
    def __init__(self, target="roots"):
        self.target = target
        self.paths = []
        self.bf = None
        
    def initialize(self):
        if not (self.target == "evidence" or self.target == "roots" or self.target == "both"):
            raise Exception("target must be 'evidence', 'roots' or 'both'")
        if (self.target == 'evidence' or self.target == 'both') and self.evidence == None:
            raise Exception("evidence cannot be None")
        
        #Initialize fallback expansion method
        self.bf = BreadthFirstExpansion()
        self.bf.set_params(model=self.model, query=self.node)
        self.bf.initialize()
        
        #Find the shortest paths to all targets (evidence, roots or both) using BFS
        targets = []
        if self.target == "roots" or self.target == "both":
            targets.extend(self.model.get_roots())
        if self.evidence != None and (self.target == "evidence" or self.target == "both"):
            targets.extend(self.evidence.keys())

        queue = [self.node]
        visited = dict(map(lambda i,j : (i,j) , self.model.nodes, [False] * len(self.model.nodes)))
        parents = dict(map(lambda i,j : (i,j) , self.model.nodes, [None] * len(self.model.nodes)))
        while not len(queue) == 0:
            node = queue.pop(0)
            visited[node] = True
            if node in targets:
                path = [node]
                parent = parents[node]
                while parent != None:
                    path.append(parent)
                    parent = parents[parent]
                self.paths.extend(reversed(path))
            
            if not (self.evidence != None and node in self.evidence):
                for neighbour in self.model.get_children(node) + self.model.get_parents(node):
                    if not visited[neighbour]:
                        queue.append(neighbour)
                        parents[neighbour] = node

    def expand(self, rate):
        #Expand along the calculated paths
        nodes_added = 0
        while not len(self.paths) == 0:
            node = self.paths.pop(0)
            if not self.model.nodes[node]['active']:
                self.model.nodes[node]['active'] = True
                nodes_added += 1
                if nodes_added == rate:
                    return nodes_added
        
        #Once all paths have been expanded upon, fall back to BF expansion
        nodes_added += self.bf.expand(rate - nodes_added)
        return nodes_added
                  
class ConstantRate():
    
    def __init__(self, constant):
        if constant <= 0 or constant > 1:
            raise Exception("The constant must be larger 0 and smaller or equal to 1")
        self.c = constant
    
    def get_rate(self, model):
        return max(1, len(model.nodes) * self.c)
    
class AdaptiveRate():
    
    def __init__(self, constant):
        self.default = constant
        self.activeset_size = 0
        self.results = []
        self.avg_interval_sizes = []
        self.rates = []

    def set_params(self, setsize, precision):
        self.active_set_size = setsize
        self.precision = precision
    
    def add_result(self, result):
        self.results.append(result)
        avg_interval_size = sum([i[0].sup - i[0].inf for i in result]) / len(result)
        self.avg_interval_sizes.append(avg_interval_size)
    
    def get_rate(self, model):
        model_size = len(model.nodes)
        if len(self.avg_interval_sizes) < 2:
            rate = int(model_size * self.default)
            self.rates.append(rate)
            self.active_set_size += rate
            return rate

        improvement = self.avg_interval_sizes[-2] - self.avg_interval_sizes[-1]
        if improvement == 0:
            rate = int(model_size * self.default)
            self.rates.append(rate)
            self.active_set_size += rate
            return rate
    
        improvement_per_node = improvement / self.rates[-1]
        improvement_needed = self.avg_interval_sizes[-1] - self.precision
        rate = max(1, int(improvement_needed / improvement_per_node))
        if rate > model_size * self.default * 3:
            rate = model_size * self.default * 3
        self.rates.append(rate)
        self.active_set_size += rate
        return rate