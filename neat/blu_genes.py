'''blu_genes.py
   Holds the gene information for a blueprint
   ---
   G. Dylan Dickerson Spring 2019
'''

# -*- coding: UTF-8 -*-
import random
from config import Config
from genes import NodeGene


class BluNodeGene(NodeGene):
    def __init__(self, id, nodetype, modPointer):
        """ A BluNodeGene encodes the basics for a blueprint node gene.
            nodetype should be "INPUT", "HIDDEN", or "OUTPUT"
            modPointer should refer to a module species id in the module
             species
        """
        self._id = id
        self._type = nodetype
        self._modPointer = modPointer

        assert(self._type in ('INPUT', 'OUTPUT', 'HIDDEN'))

    id = property(lambda self: self._id)
    type = property(lambda self: self._type)
    modPointer = property(lambda self: self._modPointer)

    def __str__(self):
        return "Node: %2d type: %6s, modulePointer: %2d" \
                %(self._id, self._type, self._modPointer)

    def get_child(self, other):
        """ Creates a new BluNodeGene ramdonly inheriting its attributes from parents """
        assert(self._id == other._id)

        ng = BluNodeGene(self._id, self._type,
                         random.choice(self._modPointer, other._modPointer))
        return ng

    def copy(self):
        return NodeGene(self._id, self._type, self._bias,
                        self._response, self._activation_type)

    def mutate(self):
        if self._type != 'HIDDEN':  # Don't modify input & output nodes
            return
        if random.random() < Config.prob_mutatemodpointer:
            self.__mutate_modpointer()

    def __mutate_modpointer(self):
        return random.choice(range(Config.modpopsize))


class BluConnectionGene(object):
    __global_innov_number = 0
    __innovations = {}  # A list of innovations.
    # Should it be global? Reset at every generation? Who knows?

    @classmethod
    def reset_innovations(cls):
        cls.__innovations = {}

    def __init__(self, innodeid, outnodeid, enabled, innov=None):
        self.__in = innodeid
        self.__out = outnodeid
        self.__enabled = enabled
        if innov is None:
            try:
                self.__innov_number = self.__innovations[self.key]
            except KeyError:
                self.__innov_number = self.__get_new_innov_number()
                self.__innovations[self.key] = self.__innov_number
        else:
            self.__innov_number = innov

    innodeid = property(lambda self: self.__in)
    outnodeid = property(lambda self: self.__out)
    enabled = property(lambda self: self.__enabled)
    # Key for dictionaries, avoids two connections between the same nodes.
    key = property(lambda self: (self.__in, self.__out))

    def mutate(self):
        r = random.random
        if r() < Config.prob_togglelink:
            self.enable()

    def enable(self):
        """ Enables a link. """
        self.__enabled = True

    @classmethod
    def __get_new_innov_number(cls):
        cls.__global_innov_number += 1
        return cls.__global_innov_number

    def __str__(self):
        s = "In %2d, Out %2d, " % (self.__in, self.__out)
        if self.__enabled:
            s += "Enabled, "
        else:
            s += "Disabled, "
        return s + "Innov %d" % (self.__innov_number,)

    def __cmp__(self, other):
        return cmp(self.__innov_number, other.__innov_number)

    def split(self, node_id):
        """ Splits a connection, creating two new connections and disabling this one """
        self.__enabled = False
        new_conn1 = BluConnectionGene(self.__in, node_id, True)
        new_conn2 = BluConnectionGene(node_id, self.__out, True)
        return new_conn1, new_conn2

    def copy(self):
        return BluConnectionGene(self.__in, self.__out,
                                 self.__enabled, self.__innov_number)

    def is_same_innov(self, cg):
        return self.__innov_number == cg.__innov_number

    def get_child(self, cg):
        return random.choice((self, cg)).copy()
