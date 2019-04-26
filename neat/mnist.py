import os
import sys
import random
from copy import deepcopy
import cPickle as pickle
import csv
from numpy import array, average
from keras_funcs import makeKeras, makeKerasGivenMods, setupMNIST, runMNIST
from config import Config, load
from blu_chromosome import Blu_Chromosome
from mod_chromosome import Mod_Chromosome
from cdn_population import CDN_Population as pop
from visualize import drawAssembled


def evaluate(bpop, mpop, trainData, num_samples, testData, datagen, epochs=10, numAssemble=25, batchSize=32,
             bluPic='', modPic='', netPic=''):
    '''The fitness of an individual is the average fitness of the network they were used in'''
    # Shuffle the training data
    data = zip(trainData[0], trainData[1])
    random.shuffle(data)
    # Split the data back into x and y data
    xtrain = array([a for a, b in data])
    ytrain = array([b for a, b in data])
    # Create a validation set from the trainData
    trainingData = (xtrain[:42500], ytrain[:42500])
    valData = (xtrain[42500:], ytrain[42500:])
    # Second entry in these values in the # of times used
    bpDict = {bp.id: [bp, 0] for bp in bpop.population}
    modDict = {mod.id: [mod, 0] for mod in mpop.population}

    for indiv in bpop:
        indiv.fitness = 0.0
    for indiv in mpop:
        indiv.fitness = 0.0

    losses = []
    accs = []
    max_acc = -1.0
    min_loss = 0.0
    bestbp = bpop.population[0]
    bestMods = []
    for i in range(numAssemble):
        bp = random.choice(bpop.population)
        if bpop.gen == -1 or bpop.gen % 10 == 0:
            # Create pictures of this generation
            model, modsUsed = makeKeras(bp, mpop, True,
                                        bluPic + "_n:" + str(i),
                                        modPic + "_n:" + str(i),
                                        netPic + "_n:" + str(i))
        else:
            # Don't create pictures
            model, modsUsed = makeKeras(bp, mpop, False)
        loss, acc = runMNIST(model, trainingData, 42500, valData, datagen, epochs, verbosity=0)
        losses.append(loss)
        accs.append(acc)
        if acc > max_acc:
            bestbp = bp
            bestMods = modsUsed
            max_acc = acc
            min_loss = loss
        bpDict[bp.id][0].fitness += acc
        bpDict[bp.id][1] += 1

        for m in modsUsed:
            modDict[m.id][0].fitness += acc
            modDict[m.id][1] += 1

    for bp, numUsed in bpDict.values():
        if numUsed > 0:
            bp.fitness /= numUsed
    for mod, numUsed in modDict.values():
        if numUsed > 0:
            mod.fitness /= numUsed

    return bestbp, bestMods, average(losses), average(accs), max_acc, min_loss


if __name__ == '__main__':
    print "\n--- Starting MNIST Experiment for CoDeepNEAT project --- \n"
    sys.stdout.flush()
    # Load the configuration file
    load('mnist_config')
    # Download the mnist dataset and setup a datagen object
    x_train, y_train, num_samples, x_test, y_test, datagen = setupMNIST(False)

    checkDir = os.path.join(os.getcwd(), "mnist/checkpoints/")
    statDir = os.path.join(os.getcwd(), "mnist/stats/")
    picDir = os.path.join(os.getcwd(), "mnist/pics/")
    pickleDir = os.path.join(os.getcwd(), "mnist/pickles/")
    if not os.path.exists(checkDir):
        os.makedirs(checkDir)
    if not os.path.exists(statDir):
        os.makedirs(statDir)
    if not os.path.exists(picDir):
        os.makedirs(picDir)
    if not os.path.exists(pickleDir):
        os.makedirs(pickleDir)
    # pickles
    bpsFile = os.path.join(pickleDir, 'blueprints.pck')
    modsFile = os.path.join(pickleDir, 'modules.pck')
    eliteBPFile = os.path.join(pickleDir, 'eliteBP.pck')
    eliteModFile = os.path.join(pickleDir, 'eliteMod.pck')
    # CSV files
    aveLossFile = os.path.join(statDir, 'aveLoss.csv')
    aveAccFile = os.path.join(statDir, 'aveAcc.csv')
    bestLossFile = os.path.join(statDir, 'bestLoss.csv')
    bestAccFile = os.path.join(statDir, 'bestAcc.csv')
    # Best network
    eliteModelFile = os.path.join(pickleDir, 'bestModel.h5')
    elitePic = os.path.join(picDir, 'eliteNetwork')

    # Create the populations of blueprints & modules
    b_pop = pop(Config.blupopsize, Blu_Chromosome)
    m_pop = pop(Config.modpopsize, Mod_Chromosome)
    # Set the evaluation functions to the one above
    b_pop.evaluate = evaluate
    m_pop.evaluate = evaluate
    m_pop.speciate()
    b_pop.speciate()

    bestBPs = []  # per generation
    bestMods = []  # per generation
    avgLosses = []
    avgAccs = []
    bestAccs = []
    bestLoss = []

    eliteBP = b_pop.population[0]
    eliteMods = []
    bestAcc = 0.0

    stepVerbosity = True

    print "\n---- Starting Evolution ----"
    sys.stdout.flush()

    # Evolve the populations
    for g in range(Config.generations):
        # Based on step, how verbose should some functions be?
        stepVerbosity = (g % 10 == 0) or (g == -1)
        # Create picture names
        bPic = os.path.join(picDir, "bp_" + str(g))
        mPic = os.path.join(picDir, "mod_" + str(g))
        nPic = os.path.join(picDir, "net_" + str(g))

        # Evaluate assembled networks
        bBP, bM, aveLoss, aveAcc, bAcc, bLoss = evaluate(b_pop, m_pop, (x_train, y_train), num_samples, (x_test, y_test), datagen,
                                                         epochs=8, numAssemble=25,
                                                         bluPic=bPic, modPic=mPic, netPic=nPic)
        # Add to record lists
        bestBPs.append(bBP)
        bestMods.append(bM)
        avgLosses.append(aveLoss)
        avgAccs.append(aveAcc)
        bestAccs.append(bAcc)
        bestLoss.append(bLoss)

        # Update elite found so far
        if bAcc > bestAcc:
            eliteBP = bBP
            eliteMods = bM
            bestAcc = bAcc

        # Evolve the populations
        if stepVerbosity:
            print "Module Step"
        endModEvol = m_pop.step(stepVerbosity, 10, False, checkDir)
        if stepVerbosity:
            print "Blueprint Step"
        endEvolBP = b_pop.step(stepVerbosity, 10, False, checkDir)
        if endModEvol or endEvolBP:
            break
        sys.stdout.flush()

    sys.stdout.flush()
    # Draw the best network
    drawBP = deepcopy(eliteBP)
    drawBP.cullDisabled()
    drawMs = []
    for m in eliteMods:
        dm = deepcopy(m)
        dm.cullDisabled()
        drawMs.append(dm)
    drawAssembled(drawBP, drawMs, elitePic)

    # Find the fitness of the best blueprint
    print "----Evolution Ended: Finding fitness of elite model----"
    eliteModel = makeKerasGivenMods(eliteBP, eliteMods, False)
    eliteLoss, eliteAcc = runMNIST(eliteModel, (x_train, y_train), num_samples, (x_test, y_test), datagen,
                                   epochs=100, verbosity=2)
    print "Elite Model loss:%f accuracy:%f" % (eliteLoss, eliteAcc)
    print "Evolution took %d generations" % g
    sys.stdout.flush()

    # Save/pickle a lot of stuff
    # Save pickels
    pickle.dump(bestBPs, open(bpsFile, 'wb'))
    pickle.dump(bestMods, open(modsFile, 'wb'))
    pickle.dump(eliteBP, open(eliteBPFile, 'wb'))
    pickle.dump(eliteMods, open(eliteModFile, 'wb'))
    #csv files
    with open(aveLossFile, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(avgLosses)
    with open(aveAccFile, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(avgAccs)
    with open(bestLossFile, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(bestLoss)
        writer.writerow([eliteLoss])
    with open(bestAccFile, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(bestAccs)
        writer.writerow([eliteAcc])
    # model
    eliteModel.save(eliteModFile)
