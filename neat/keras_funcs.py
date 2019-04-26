from math import ceil
import random
from copy import deepcopy
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model
from keras.datasets import mnist

from config import Config, load
from visualize import draw_module, draw_blu, drawAssembled


def makeKeras(blueprint, module_pop, visualize=False, bluFile="", modFile="", netFile=""):
    '''Assumes no skip connections in blueprint or modules'''
    # Start with defining the input layer
    main_input = Input(shape=Config.input_nodes, name='main_input')
    if visualize:
        drawblu = deepcopy(blueprint)
        drawblu.cullDisabled()
        draw_blu(drawblu, bluFile + "_" + str(blueprint.id))
    modsUsed = []

    nodes = blueprint.node_genes
    blu_nodeDict = {ng.id: ng for ng in nodes}
    conns = blueprint.conn_genes
    blu_connsDict = {cg.innodeid: cg for cg in conns if cg.enabled}
    mod_species = module_pop.species
    speciesDict = {s.id: s.members for s in mod_species}

    blu_conn = blu_connsDict[1]
    b_inID = blu_conn.outnodeid
    while b_inID != 2:  # While not to the output node
        blu_node = blu_nodeDict[b_inID]
        module_C = random.choice(speciesDict[blu_node.modPointer])
        if visualize:
            drawmod = deepcopy(module_C)
            modsUsed.append(drawmod)
            drawmod.cullDisabled()
            draw_module(drawmod, modFile + "_" + str(blu_node.modPointer) + "_" + str(module_C.id))
        m_inID = 1
        m_nodeDict = {mg.id: mg for mg in module_C.node_genes}
        m_connDict = {cmg.innodeid: cmg for cmg in module_C.conn_genes}
        mod = m_nodeDict[m_inID]
        if len(module_C.node_genes) > 1:  # if there are connections
            m_inID = m_connDict[m_inID].outnodeid
        else:
            m_inID = 2
        x = Conv2D(mod.layersize, (mod.kernel_size, mod.kernel_size),
                   strides=(1, 1), padding=mod.padding, activation=mod.activation_type)(main_input)
        if mod.maxpool:
            x = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(x)
        x = Dropout(mod.dropout)(x)
        while m_inID != 2:
            if len(module_C.node_genes) > 1:  # if there are connections
                mod = m_nodeDict[m_inID]
                m_inID = m_connDict[m_inID].outnodeid
                x = Conv2D(mod.layersize, (mod.kernel_size, mod.kernel_size),
                           strides=(1, 1), padding=mod.padding, activation=mod.activation_type)(x)
                if mod.maxpool:
                    x = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(x)
                x = Dropout(mod.dropout)(x)  # use 1-droprate b/c tensorflow is apprently using keep_prob instead
            else:
                m_inID = 2

        blu_conn = blu_connsDict[b_inID]
        b_inID = blu_conn.outnodeid

    if visualize:
        drawBP = deepcopy(blueprint)
        drawBP.cullDisabled()
        drawMs = []
        for m in modsUsed:
            dm = deepcopy(m)
            dm.cullDisabled()
            drawMs.append(dm)
        drawAssembled(drawBP, drawMs, netFile + '_blu:' + str(blueprint.id))

    # Also define the output layer
    x = Flatten()(x)
    main_output = Dense(Config.output_nodes, activation='softmax', name='main_output')(x)

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model, modsUsed


def makeKerasMostFit(blueprint, modPop, visualize=False, bluFile="", modFile="", netFile=""):
    ''' Find the best modules in modPop and make a network with the best modules this blueprint uses'''
    pass


def makeKerasGivenMods(blueprint, mods, visualize=False, bluFile="", modFile="", netFile=""):
    main_input = Input(shape=Config.input_nodes, name='main_input')
    if visualize:
        drawblu = deepcopy(blueprint)
        drawblu.cullDisabled()
        draw_blu(drawblu, bluFile)

    nodes = blueprint.node_genes
    blu_nodeDict = {ng.id: ng for ng in nodes}
    conns = blueprint.conn_genes
    blu_connsDict = {cg.innodeid: cg for cg in conns if cg.enabled}

    blu_conn = blu_connsDict[1]
    b_inID = blu_conn.outnodeid
    for module_C in mods:
        blu_node = blu_nodeDict[b_inID]
        if visualize:
            drawmod = deepcopy(module_C)
            modsUsed.append(drawmod)
            drawmod.cullDisabled()
            draw_module(drawmod, modFile + "_" + str(blu_node.modPointer) + "_" + str(module_C.id))
        m_inID = 1
        m_nodeDict = {mg.id: mg for mg in module_C.node_genes}
        m_connDict = {cmg.innodeid: cmg for cmg in module_C.conn_genes}
        mod = m_nodeDict[m_inID]
        if len(module_C.node_genes) > 1:  # if there are connections
            m_inID = m_connDict[m_inID].outnodeid
        else:
            m_inID = 2
        x = Conv2D(mod.layersize, (mod.kernel_size, mod.kernel_size),
                   strides=(1, 1), padding=mod.padding, activation=mod.activation_type)(main_input)
        if mod.maxpool:
            x = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(x)
        x = Dropout(mod.dropout)(x)
        while m_inID != 2:
            if len(module_C.node_genes) > 1:  # if there are connections
                mod = m_nodeDict[m_inID]
                m_inID = m_connDict[m_inID].outnodeid
                x = Conv2D(mod.layersize, (mod.kernel_size, mod.kernel_size),
                           strides=(1, 1), padding=mod.padding, activation=mod.activation_type)(x)
                if mod.maxpool:
                    x = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(x)
                x = Dropout(mod.dropout)(x)  # use 1-droprate b/c tensorflow is apprently using keep_prob instead
            else:
                m_inID = 2

        blu_conn = blu_connsDict[b_inID]
        b_inID = blu_conn.outnodeid

    #if visualize:
    #    drawAssembled(blueprint, modsUsed, netFile)

    # Also define the output layer
    x = Flatten()(x)
    main_output = Dense(Config.output_nodes, activation='softmax', name='main_output')(x)

    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


def setupMNIST(verbose=False):
    num_classes = Config.output_nodes
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if verbose:
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    # Put the inputs into a more usable range
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # use channels last convention
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Setup a data generator
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    datagen.fit(x_train)

    return x_train, y_train, x_train.shape[0], x_test, y_test, datagen


def runMNIST(model, trainData, num_samples, valData, datagen, epochs=10, batchSize=32, verbosity=0):
    spe = ceil(num_samples / batchSize)
    model.fit_generator(datagen.flow(trainData[0], trainData[1], batchSize),
                        steps_per_epoch=ceil(num_samples / batchSize),
                        epochs=epochs,
                        validation_data=(valData[0], valData[1]),
                        verbose=verbosity,
                        workers=4)

    (loss, accuracy) = model.evaluate(valData[0], valData[1])

    return loss, accuracy


if __name__ == '__main__':
    from cdn_population import CDN_Population as pop
    from blu_chromosome import Blu_Chromosome
    from mod_chromosome import Mod_Chromosome

    print "Testing keras_functions with generous configuration parameters"

    load('template_config')
    Config.input_nodes = [int(28), int(28), int(1)]
    Config.output_nodes = 10
    Config.max_fitness_threshold = 2
    Config.prob_addmodule = 0.0
    Config.prob_addconn = 0.03
    Config.min_learnrate = 0.001
    Config.max_learnrate = 0.1
    Config.prob_mutatelearnrate = 1
    Config.learnrate_mutation_power = 0.002
    Config.min_size = 3
    Config.max_size = 256
    Config.size_mutation_power = 5
    Config.min_ksize = 3
    Config.max_ksize = 5
    Config.start_drop = 0.2
    Config.min_drop = 0.0
    Config.max_drop = 0.7
    Config.drop_mutation_power = 0.05
    Config.prob_addlayer = 0.5
    Config.prob_mutatelayersize = 0.5
    Config.prob_mutatekernel = 0.5
    Config.prob_mutatepadding = 0.5
    Config.prob_mutatedrop = 0.5
    Config.prob_mutatemaxpool = 0.5

    # sample fitness function
    def eval_fitness(population):
        for individual in population.population:
            individual.fitness = 1.0

    pop.evaluate = eval_fitness

    b_pop = pop(Config.blupopsize, Blu_Chromosome)
    m_pop = pop(Config.modpopsize, Mod_Chromosome)

    for i in range(21):
        m_pop.evaluate()
        b_pop.evaluate()
        m_pop.step(False)
        b_pop.step(False)

    bp = b_pop.population[0]

    model, _ = makeKeras(bp, m_pop, True, "pics/makeKeras_blu", "pics/makeKeras_mod", "pics/assembledNet")
    x_train, y_train, num_samples, x_test, y_test, datagen = setupMNIST(True)
    loss, accuracy = runMNIST(model, (x_train, y_train), num_samples, (x_test, y_test), datagen, 2)

    print "loss: %f\tacc: %f"%(loss, accuracy)
