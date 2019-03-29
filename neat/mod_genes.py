# -*- coding: UTF-8 -*-
import random
from config import Config
from itertools import product


class ModNodeGene(object):
    def __init__(self, id, nodetype, layersize, activation_type=None):
        """ A node gene encodes the basic artificial neuron model.
            nodetype should be "INPUT", "HIDDEN", or "OUTPUT"
        """
        self._id = id
        self._type = nodetype
        self._layersize = layersize
        self._activation_type = activation_type

        assert(self._type in ('INPUT', 'OUTPUT', 'HIDDEN'))
        assert(self._activation_type in ('tanh', 'relu', 'sigmoid'))

    id = property(lambda self: self._id)
    type = property(lambda self: self._type)
    layersize = property(lambda self: self._layersize)
    activation_type = property(lambda self: self._activation_type)

    def __str__(self):
        return "Node %2d %6s, size %d" \
                %(self._id, self._type, self._layersize)

    def get_child(self, other):
        """ Creates a new ModNodeGene ramdonly inheriting its attributes from parents """
        assert(self._id == other._id)

        ng = ModNodeGene(self._id, self._type,
                         random.choice((self._layersize, other._layersize)),
                         random.choice((self._activation_type,
                                        other._activation_type)))
        return ng

    def __mutate_size(self):
        self._layersize += random.gauss(0, 1) * Config.size_mutation_power
        if self._layersize > Config.max_size:
            self._layersize = Config.max_size
        elif self._layersize < Config.min_size:
            self._layersize = Config.min_size

    def __mutate_activation(self):
        self._activation_type = random.choice(('tanh', 'relu', 'sigmoid'))

    def copy(self):
        return ModNodeGene(self._id, self._type,
                           self._layersize, self._activation_type)

    def mutate(self):
        r = random.random
        if r() < Config.prob_mutatesize:
            self.__mutate_size()
        if r() < Config.prob_mutateactivation:
            self.__mutate_activation()


class ConvModGene(ModNodeGene):
    def __init__(self, id, nodetype, numfilts, activation_type, k_size, strides,
                 padding, dropout, maxpool, batchnorm):

        self._id = id
        self._type = nodetype
        self._layersize = numfilts
        self._activation_type = activation_type
        self._kernel_size = k_size
        self._strides = strides
        self._padding = padding
        self._dropout = dropout
        self._maxpool = maxpool
        self._batchnorm = batchnorm

        assert(self._layersize in range(Config.min_filters, Config.max_filters))
        assert(self._kernel_size in range(Config.min_ksize, Config.max_ksize))
        assert(self._strides in range(Config.min_stride, Config.max_stride))
        assert(self._padding in ('same', 'valid'))
        assert(self._dropout in range(Config.min_drop, Config.max_drop,
                                      Config.drop_mutsplit))
        assert(self._maxpool in (True, False))
        assert(self._batchnorm in (True, False))

    # Getters for values, setters are done through mutate
    kernel_size = property(lambda self: self._kernel_size)
    strides = property(lambda self: self._strides)
    padding = property(lambda self: self._padding)
    dropout = property(lambda self: self._dropout)
    maxpool = property(lambda self: self._maxpool)
    batchnorm = property(lambda self: self._batchnorm)

    # Mutators. analagous to setters
    # TODO: Implement mutators
    def _mutate_kernel(self):
        self._kernel_size = random.randint(Config.min_ksize, Config.max_ksize)

    def _mutate_strides(self):
        self._strides = random.randint(Config.min_stride, Config.max_stride)

    def _mutate_padding(self):
        self._padding = random.choice(('same', 'valid'))

    def _mutate_dropout(self):
        self._dropout += random.gauss(0, 1) * Config.drop_mutation_power
        if self._dropout < Config.min_drop:
            self.dropout = Config.min_drop
        if self._dropout > Config.max_drop:
            self._dropout = Config.max_drop

    def _mutate_maxpool(self):
        self._maxpool = random.choice((True, False))

    def _mutate_batchnorm(self):
        self._batchnorm = random.choice((True, False))

    def mutate(self):
        super(ConvModGene, self).mutate()
        r = random.random
        if r() < Config.prob_mutatelayersize:
            self._mutate_size()
        if r() < Config.prob_mutateactivation:
            self.__mutate_activation
        if r() < Config.prob_mutatekernel:
            self._mutate_kernel()
        if r() < Config.prob_mutatestride:
            self._mutate_strides()
        if r() < Config.prob_mutatepadding:
            self._mutate_padding()
        if r() < Config.prob_mutatedrop:
            self._mutate_dropout()
        if r() < Config._mutate_maxpool:
            self._mutate_maxpool()
        if r() < Config.prob_mutatebatch:
            self._mutate_batchnorm()

    def get_child(self):
        assert(self.id == self.id)

        ng = ConvModGene(self._id, self._layersize, self._activation_type,
                         self._kernel_size, self._strides, self._padding,
                         self._dropout, self._maxpool, self._batchnorm)
        return ng

    def copy(self):
        return ConvModGene(self._id, self._layersize, self._activation_type,
                           self._kernel_size, self._strides, self._padding,
                           self._dropout, self._maxpool, self._batchnorm)


class ModConnectionGene(object):
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
        new_conn1 = ModConnectionGene(self.__in, node_id, True)
        new_conn2 = ModConnectionGene(node_id, self.__out, True)
        return new_conn1, new_conn2

    def copy(self):
        return ModConnectionGene(self.__in, self.__out,
                                 self.__enabled, self.__innov_number)

    def is_same_innov(self, cg):
        return self.__innov_number == cg.__innov_number

    def get_child(self, cg):
        return random.choice((self, cg)).copy()
