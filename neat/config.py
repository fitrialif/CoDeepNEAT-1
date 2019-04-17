# sets the configuration parameters for NEAT
from ConfigParser import ConfigParser


def load(file):
    try:
        config_file = open(file, 'r')
    except IOError:
        print 'Error: file %s not found!' %file
        raise
    else:
        parameters = ConfigParser()
        parameters.readfp(config_file)

        # set class attributes
        # phenotype
        Config.input_nodes = str(parameters.get('phenotype', 'input_nodes'))
        Config.input_nodes = list(int(x) for x in Config.input_nodes.split(','))
        Config.output_nodes = int(parameters.get('phenotype', 'output_nodes'))
        Config.hidden_nodes = int(parameters.get('phenotype', 'hidden_nodes'))

        # Blueprint
        Config.min_learnrate = float(parameters.get('blueprint', 'min_learnrate'))
        Config.max_learnrate = float(parameters.get('blueprint', 'max_learnrate'))
        Config.learnrate_mutation_power = float(parameters.get('blueprint', 'learnrate_mutation_power'))
        #Config.min_momentum = float(parameters.get('blueprint', 'min_momentum'))
        #Config.max_momentum = float(parameters.get('blueprint', 'max_momentum'))
        #Config.momentum_mutation_power = float(parameters.get('blueprint', 'momentum_mutation_power'))
        #Config.min_cropsize = int(parameters.get('blueprint', 'min_cropsize'))
        #Config.max_cropsize = int(parameters.get('blueprint', 'max_cropsize'))
        #Config.cropsize_mutation_power = int(parameters.get('blueprint', 'cropsize_mutation_power'))
        Config.prob_mutatemodpointer = float(parameters.get('blueprint', 'prob_mutatemodpointer'))
        Config.prob_mutatelearnrate = float(parameters.get('blueprint', 'prob_mutatelearnrate'))
        #Config.prob_mutatemomentum = float(parameters.get('blueprint', 'prob_mutatemomentum'))
        #Config.prob_mutatecropsize = float(parameters.get('blueprint', 'prob_mutatecropsize'))
        #Config.prob_mutatehorizontalflips = float(parameters.get('blueprint', 'prob_mutatehorizontalflips'))

        # Module
        Config.min_size = int(parameters.get('module', 'min_size'))
        Config.max_size = int(parameters.get('module', 'max_size'))
        Config.size_mutation_power = float(parameters.get('module', 'size_mutation_power'))
        Config.min_ksize = int(parameters.get('module', 'min_ksize'))
        Config.max_ksize = int(parameters.get('module', 'max_ksize'))
        Config.min_stride = int(parameters.get('module', 'min_stride'))
        Config.max_stride = int(parameters.get('module', 'max_stride'))
        Config.min_drop = float(parameters.get('module', 'min_drop'))
        Config.max_drop = float(parameters.get('module', 'max_drop'))
        Config.drop_mutation_power = float(parameters.get('module', 'drop_mutation_power'))
        Config.prob_mutatelayersize = float(parameters.get('module', 'prob_mutatelayersize'))
        Config.prob_mutateactivation = float(parameters.get('module', 'prob_mutateactivation'))
        Config.prob_mutatekernel = float(parameters.get('module', 'prob_mutatekernel'))
        Config.prob_mutatestride = float(parameters.get('module', 'prob_mutatestride'))
        Config.prob_mutatepadding = float(parameters.get('module', 'prob_mutatepadding'))
        Config.prob_mutatedrop = float(parameters.get('module', 'prob_mutatedrop'))
        Config.prob_mutatemaxpool = float(parameters.get('module', 'prob_mutatemaxpool'))
        Config.prob_mutatebatchnorm = float(parameters.get('module', 'prob_mutatebatchnorm'))


        # GA
        Config.pop_size = int(parameters.get('genetic', 'pop_size'))
        Config.max_fitness_threshold = float(parameters.get('genetic','max_fitness_threshold'))
        Config.prob_addconn = float(parameters.get('genetic', 'prob_addconn'))
        Config.prob_addnode = float(parameters.get('genetic', 'prob_addnode'))
        Config.elitism = float(parameters.get('genetic', 'elitism'))

        # genotype compatibility
        Config.compatibility_threshold = float(parameters.get('genotype compatibility', 'compatibility_threshold'))
        Config.compatibility_change = float(parameters.get('genotype compatibility', 'compatibility_change'))
        Config.excess_coeficient = float(parameters.get('genotype compatibility', 'excess_coeficient'))
        Config.disjoint_coeficient = float(parameters.get('genotype compatibility', 'disjoint_coeficient'))
        Config.weight_coeficient = float(parameters.get('genotype compatibility', 'weight_coeficient'))

        # species
        Config.species_size = int(parameters.get('species', 'species_size'))
        Config.survival_threshold = float(parameters.get('species', 'survival_threshold'))
        Config.old_threshold = int(parameters.get('species', 'old_threshold'))
        Config.youth_threshold = int(parameters.get('species', 'youth_threshold'))
        Config.old_penalty = float(parameters.get('species', 'old_penalty'))    # always in (0, 1)
        Config.youth_boost = float(parameters.get('species', 'youth_boost'))   # always in (1, 2)
        Config.max_stagnation = int(parameters.get('species', 'max_stagnation'))


class Config:

    # phenotype config
    input_nodes = None
    output_nodes = None
    hidden_nodes = None

    # Blueprint config
    #  Except for modpointer, these are global parameters (i.e. in chromosome)
    min_learnrate = None
    max_learnrate = None
    learnrate_mutation_power = None
    min_momentum = None
    max_momentum = None
    momentum_mutation_power = None
    min_cropsize = None
    max_cropsize = None
    cropsize_mutation_power = None
    prob_mutatemodpointer = None
    prob_mutatelearnrate = None
    prob_mutatemomentum = None
    prob_mutatecropsize = None
    prob_mutatehorizontalflips = None


    # Module Config
    #  Number of filters
    min_size = None
    max_size = None
    size_mutation_power = None
    #  Kernel size
    min_ksize = None
    max_ksize = None
    # Stride size
    min_stride = None
    max_stride = None
    # Dropout
    min_drop = None
    max_drop = None
    drop_mutation_power = None
    # Probabilities of changing values
    prob_mutatelayersize = None
    prob_mutateactivation = None
    prob_mutatekernel = None
    prob_mutatestride = None
    prob_mutatepadding = None
    prob_mutatedrop = None
    prob_mutatemaxpool = None
    prob_mutatebatchnorm = None

    # GA config
    pop_size = None
    max_fitness_threshold = None
    prob_addconn = None
    prob_addnode = None
    prob_togglelink = None
    elitism = None

    # genotype compatibility
    compatibility_threshold = None
    compatibility_change = None
    excess_coeficient = None
    disjoint_coeficient = None
    weight_coeficient = None

    # species
    species_size = None
    survival_threshold = None  # only the best 20% for each species is allowed to mate
    old_threshold = None
    youth_threshold = None
    old_penalty = None    # always in (0, 1)
    youth_boost = None    # always in (1, 2)
    max_stagnation = None
