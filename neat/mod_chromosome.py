import random
import math
from config import Config
import genome

# Temporary workaround - default settings
#node_gene_type = genome.NodeGene
conn_gene_type = genome.ConnectionGene


class Mod_Chromosome(FFChromosome):
    """ A chromosome for general recurrent neural networks. """
    _id = 0

    def __init__(self, parent1_id, parent2_id, node_gene_type, conn_gene_type):
        pass

    @classmethod
    def __get_new_id(cls):
        cls._id += 1
        return cls._id

    def mutate(self):
        pass

    def crossover(self, other):
        pass

    def _inherit_genes(child, parent1, parent2):
        pass

    def _mutate_add_node(self):
        pass

    def _mutate_add_connection(self):
        pass

    def distance(self, other):
        pass

    def size(self):
        pass

    def __cmp__(self, other):
        pass

    def __str__(self):
        pass

    @classmethod
    def create_minimal_model(cls):
        pass


if __name__ == '__main__':
    # Example
    import visualize
    # define some attributes
    node_gene_type = genome.NodeGene         # standard neuron model
    conn_gene_type = genome.ConnectionGene   # and connection link
    Config.nn_activation = 'exp'             # activation function
    Config.weight_stdev = 0.9                # weights distribution

    Config.input_nodes = 2                   # number of inputs
    Config.output_nodes = 1                  # number of outputs

    # creates a chromosome for recurrent networks
    #c1 = Chromosome.create_fully_connected()

    # creates a chromosome for feedforward networks
    c2 = FFChromosome.create_fully_connected()
    # add two hidden nodes
    c2.add_hidden_nodes(2)
    # apply some mutations
    # c2._mutate_add_node()
    # c2._mutate_add_connection()

    # check the result
    # visualize.draw_net(c1) # for recurrent nets
    visualize.draw_ff(c2)   # for feedforward nets
    # print the chromosome
    print c2
