import random
import math
from copy import deepcopy
from config import Config, load
import genes
from chromosome import FFChromosome
from blu_genes import BluNodeGene, BluConnectionGene


class Blu_Chromosome(FFChromosome):
    """ A chromosome for a CoDeepNEAT blueprint """
    _id = 0

    def __init__(self, parent1_id, parent2_id,
                 node_gene_type=BluNodeGene, conn_gene_type=BluConnectionGene):
        super(Blu_Chromosome, self).__init__(parent1_id, parent2_id, node_gene_type, conn_gene_type)
        self.learnrate = Config.min_learnrate
        self.__node_order = []

    inputs = property(lambda self: self._input_nodes)
    outputs = property(lambda self: self._output_nodes)
    node_order = property(lambda self: self.__node_order)

    def mutate(self):
        """ Mutate this blueprint
            Either add a module, add a connection, or mutate nodes & connections
        """
        r = random.random
        if r() < Config.prob_mutatelearnrate:
            self._mutate_learnrate()
        if r() < Config.prob_addmodule:
            self._mutate_add_node()
        elif r() < Config.prob_addconn:
            self._mutate_add_connection()
        else:
            for cg in self._connection_genes.values():
                cg.mutate()
            for ng in self._node_genes:
                ng.mutate()

    # Use crossover function from Chromosome
    # def crossover(self, other)

    def _inherit_genes(child, parent1, parent2):
        """ Applies the crossover operator. """
        super(FFChromosome, child)._inherit_genes(parent1, parent2)

        # Crossover the learnrate from the fittest parent
        child.learnrate = parent1.learnrate

    def _mutate_add_node(self):
        # Choose a random connection to split
        conn_to_split = random.choice(self._connection_genes.values())
        # Add a new neuron that points to a random module species
        ng = self._node_gene_type(len(self._node_genes) + 1, 'HIDDEN',
                                  random.randint(1, Config.modpopsize))
        self._node_genes.append(ng)
        new_conn1, new_conn2 = conn_to_split.split(ng.id)
        self._connection_genes[new_conn1.key] = new_conn1
        self._connection_genes[new_conn2.key] = new_conn2
        print "_mutate_add_node"
        print self.__node_order
        # Add node to node order list: after the presynaptic node of the split connection
        # and before the postsynaptic node of the split connection
        if self._node_genes[conn_to_split.innodeid - 1].type == 'HIDDEN':
            mini = self.__node_order.index(conn_to_split.innodeid) + 1
        else:
            # Presynaptic node is an input node, not hidden node
            mini = 0
        if self._node_genes[conn_to_split.outnodeid - 1].type == 'HIDDEN':
            maxi = self.__node_order.index(conn_to_split.outnodeid)
        else:
            # Postsynaptic node is an output node, not hidden node
            maxi = len(self.__node_order)
        self.__node_order.insert(random.randint(mini, maxi), ng.id)
        assert(len(self.__node_order) == len([n for n in self.node_genes if n.type == 'HIDDEN']))
        return (ng, conn_to_split)

    def _mutate_add_connection(self):
        # Only for feedforwad networks
        num_hidden = len(self.__node_order)
        num_output = 1  # There are only one input and ouput modules
        num_input = 1

        total_possible_conns = (num_hidden + num_output) * (num_input + num_hidden) - \
            sum(range(num_hidden + 1))

        remaining_conns = total_possible_conns - len(self._connection_genes)
        # Check if new connection can be added:
        if remaining_conns > 0:
            n = random.randint(0, remaining_conns - 1)
            count = 0
            # Count connections
            for in_node in (self._node_genes[:num_input] + self._node_genes[-num_hidden:]):
                for out_node in self._node_genes[1:]:
                    if (in_node.id, out_node.id) not in self._connection_genes.keys() and \
                        self.__is_connection_feedforward(in_node, out_node):
                        # Free connection
                        if count == n:  # Connection to create
                            cg = self._conn_gene_type(in_node.id, out_node.id, True)
                            self._connection_genes[cg.key] = cg
                            return
                        else:
                            count += 1

    def __is_connection_feedforward(self, in_node, out_node):
        return in_node.type == 'INPUT' or out_node.type == 'OUTPUT' or \
            self.__node_order.index(in_node.id) < self.__node_order.index(out_node.id)

    def _mutate_learnrate(self):
        self.learnrate += random.gauss(0, 1) * Config.learnrate_mutation_power
        self.learnrate = round(self.learnrate, 5)
        if self.learnrate > Config.max_learnrate:
            self.learnrate = Config.max_learnrate
        elif self.learnrate < Config.min_learnrate:
            self.learnrate = Config.min_learnrate

    def distance(self, other):
        """ Returns the distance between this chromosome and the other. """
        if len(self._connection_genes) > len(other._connection_genes):
            chromo1 = self
            chromo2 = other
        else:
            chromo1 = other
            chromo2 = self

        matching = 0
        disjoint = 0
        excess = 0

        max_cg_chromo2 = max(chromo2._connection_genes.values())

        for cg1 in chromo1._connection_genes.values():
            try:
                cg2 = chromo2._connection_genes[cg1.key]
            except KeyError:
                if cg1 > max_cg_chromo2:
                    excess += 1
                else:
                    disjoint += 1
            else:
                # Homologous genes
                matching += 1

        disjoint += len(chromo2._connection_genes) - matching

        # assert(matching > 0) # this can't happen
        distance = Config.excess_coeficient * excess + \
                   Config.disjoint_coeficient * disjoint

        return distance

    def size(self):
        # Number of hidden nodes
        num_hidden = len(self.__node_order)
        # Number of enabled connections
        conns_enabled = sum([1 for cg in self._connection_genes.values() if cg.enabled is True])
        return (num_hidden, conns_enabled)

    def cullDisabled(self):
        self._connection_genes = {k: cg for k, cg in self._connection_genes.iteritems() if cg.enabled}


    @classmethod
    def create_minimal_blueprint(cls):
        c = cls(0, 0, node_gene_type, conn_gene_type)
        id = 1
        # Create input node
        c._node_genes.append(c._node_gene_type(id, 'INPUT', -1))
        id += 1
        # Create output node
        c._node_genes.append(c._node_gene_type(id, 'OUTPUT', -2))
        id += 1
        # Add one hidden node
        h = c._node_gene_type(id, 'HIDDEN', random.randint(1, Config.modpopsize))
        c._node_genes.append(h)
        c.__node_order.append(h.id)
        id += 1

        # Connect the input to the hidden and the hidden to the output
        intohidd = c._conn_gene_type(1, 3, True)
        c._connection_genes[intohidd.key] = intohidd
        hiddtoout = c._conn_gene_type(3, 2, True)
        c._connection_genes[hiddtoout.key] = hiddtoout

        return c

    def __str__(self):
        s = 'learnrate: '
        s += str(self.learnrate) + '\n'
        s += "Nodes:"
        for ng in self._node_genes:
            s += "\n\t" + str(ng)
        s += "\nConnections:"
        connections = self._connection_genes.values()
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)
        return s


if __name__ == '__main__':
    # Example
    import visualize
    # define some attributes
    node_gene_type = BluNodeGene        # standard neuron model
    conn_gene_type = BluConnectionGene   # and connection link
    load('template_config')
    Config.input_nodes = [32, 32, 3]          # number of inputs
    Config.output_nodes = 10                  # number of outputs
    Config.modpopsize = 10
    Config.prob_addmodule = 0.5
    Config.prob_addconn = 0.03
    Config.min_learnrate = 0.001
    Config.max_learnrate = 0.1
    Config.prob_mutatelearnrate = 1
    Config.learnrate_mutation_power = 0.002

    # creates a chromosome for recurrent networks
    c = Blu_Chromosome.create_minimal_blueprint()
    c1 = deepcopy(c)
    # check the result
    visualize.draw_blu(c, 'blu_premutate')   # for feedforward nets
    # print the chromosome
    print "intial genome"
    print str(c)

    for i in range(15):
        c.mutate()
    c2 = deepcopy(c)
    visualize.draw_blu(c, 'blu_postmutate')
    print "post mutation genome"
    print str(c)

    c.cullDisabled()
    print "pruned disabled"
    print str(c)
    c3 = deepcopy(c)

    print "Distances\n\tPremutate vs Post"
    s = "\t" + str(c.distance(c1))
    print s
    print "\tPost vs Post"
    s = "\t" + str(c2.distance(c2))
    print s
    print "\tPost vs Culled"
    s = "\t" + str(c3.distance(c2))
    print s
