import random
import math
from copy import deepcopy
from config import Config, load
from chromosome import FFChromosome
from mod_genes import ModNodeGene, ConvModGene, ModConnectionGene

# Temporary workaround - default settings
# node_gene_type = genome.NodeGene
# conn_gene_type = genome.ConnectionGene


class Mod_Chromosome(FFChromosome):
    """ A chromosome for general recurrent neural networks. """
    _id = 0

    def __init__(self, parent1_id, parent2_id,
                 node_gene_type=ModConnectionGene, conn_gene_type=ModConnectionGene):
        super(Mod_Chromosome, self).__init__(parent1_id, parent2_id, node_gene_type, conn_gene_type)
        self.__node_order = []

    node_gene_type = property(lambda self: self._node_gene_type)

    @classmethod
    def __get_new_id(cls):
        cls._id += 1
        return cls._id

    def mutate(self):
        r = random.random
        # Add a node
        if r() < Config.prob_addlayer:
            self._mutate_add_node()
        else:
            for ng in self._node_genes:
                ng.mutate()

    # Use crossover function from Chromosome
    # def crossover(self, other)

    def _inherit_genes(child, parent1, parent2):
        super(Mod_Chromosome, child)._inherit_genes(parent1, parent2)

    def _mutate_add_node(self):
        if len(self._node_genes) == 1:
            ig = self._node_genes[0]
            # Add a nother node "downstream for the only node"
            if self._node_gene_type == ConvModGene:
                ng = self._node_gene_type(len(self._node_genes) + 1, 'OUTPUT', Config.min_size,
                                          'relu', Config.min_ksize, None, 'same', Config.min_drop,
                                          True, None)
            elif self._node_gene_type == ModNodeGene:
                ng = self._node_gene_type(len(self._node_genes) + 1, 'OUTPUT',
                                          random.randint(1, Config.modpopsize), 'relu')
            self._node_genes.append(ng)
            new_conn = ModConnectionGene(ig.id, ng.id, True)
            self._connection_genes[new_conn.key] = new_conn
            return (ng, new_conn)
        else:
            # Choose a random connection to split
            conn_to_split = random.choice(self._connection_genes.values())
            #while not conn_to_split.enabled:
            #    conn_to_split = random.choice(self._connection_genes.values())
            # Add a new neuron that points to a random module species
            if self._node_gene_type == ConvModGene:
                ng = self._node_gene_type(len(self._node_genes) + 1, 'HIDDEN', Config.min_size,
                                          'relu', Config.min_ksize, None, 'same', Config.min_drop,
                                          True, None)
            elif self._node_gene_type == ModNodeGene:
                ng = self._node_gene_type(len(self._node_genes) + 1, 'HIDDEN',
                                          random.randint(1, Config.modpopsize), 'relu')
            self._node_genes.append(ng)
            new_conn1, new_conn2 = conn_to_split.split(ng.id)
            self._connection_genes[new_conn1.key] = new_conn1
            self._connection_genes[new_conn2.key] = new_conn2
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
        pass

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

        if len(chromo2._connection_genes.values()) > 0:
            max_cg_chromo2 = max(chromo2._connection_genes.values())
        else:
            max_cg_chromo2 = self._conn_gene_type(-2, -2, False, -1)

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
        # Number of enable connections
        conns_enabled = sum([1 for cg in self._connection_genes.values() if cg.enabled])
        return (num_hidden, conns_enabled)

    def __str__(self):
        s = "Nodes:"
        for ng in self._node_genes:
            s += "\n\t" + str(ng)
        s += "\nConnections:"
        connections = self._connection_genes.values()
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)
        return s

    @classmethod
    def create_minimal_module(cls):
        c = cls(0, 0, node_gene_type, conn_gene_type)
        id = 1
        # Create an input node
        c._node_genes.append(ConvModGene(id, 'INPUT', Config.min_size, 'relu', Config.min_ksize, Config.min_stride,
                                         'same', Config.min_drop, False, False))
        id += 1
        # Create output node
        # c._node_genes.append(ConvModGene(id, 'OUTPUT', Config.min_size, 'relu', Config.min_ksize, Config.min_stride,
        #                                 'same', Config.min_drop, False, False))
        # id += 1

        # Connect the input to the output
        # intoout = c._conn_gene_type(1, 2, True)
        # c._connection_genes[intoout.key] = intoout
        return c

    def cullDisabled(self):
        self._connection_genes = {k: cg for k, cg in self._connection_genes.iteritems() if cg.enabled}


if __name__ == '__main__':
    # Example
    import visualize
    # define some attributes
    node_gene_type = ConvModGene         # standard neuron model
    conn_gene_type = ModConnectionGene   # and connection link

    print "Testing module chromosome functions with excessive configuration parameters"

    # Necessary config values
    load('template_config')
    Config.min_size = 32
    Config.max_size = 256
    Config.size_mutation_power = 3
    Config.min_ksize = 3
    Config.max_ksize = 5
    Config.min_drop = 0.0
    Config.max_drop = 0.7
    Config.drop_mutation_power = 0.05
    Config.prob_addlayer = 1
    Config.prob_mutatelayersize = 0.5
    Config.prob_mutatekernel = 0.5
    Config.prob_mutatepadding = 0.5
    Config.prob_mutatedrop = 0.5
    Config.prob_mutatemaxpool = 0.5

    c = Mod_Chromosome.create_minimal_module()
    c1 = deepcopy(c)
    visualize.draw_module(c, 'mod_premutate')
    print "initial geneome"
    print str(c)

    for i in range(15):
        c.mutate()

    visualize.draw_module(c, 'mod_postmutate')
    print "post mutate geneome"
    print str(c)
    c2 = deepcopy(c)

    c.cullDisabled()
    visualize.draw_module(c, "mod_post_cull")
    print "post cull"
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
