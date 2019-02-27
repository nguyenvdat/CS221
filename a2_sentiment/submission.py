#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    word_count = {}
    for word in x.split():
        word_count[word] = word_count.get(word, 0) + 1
    return word_count 
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    def predictor(x):
        if dotProduct(featureExtractor(x), weights) > 0:
            return 1
        else:
            return -1
    train_index = 0
    for iter in range(0, len(trainExamples) * numIters):
        train_index = random.randint(0, len(trainExamples) - 1)
        x, y = trainExamples[train_index]
        features = featureExtractor(x)
        grad = {}
        if 1 - dotProduct(weights, features) * y > 0:
            increment(grad, -y, features)
        increment(weights, -eta, grad)
        # print(evaluatePredictor(trainExamples, predictor))
        #print(evaluatePredictor(testExamples, predictor))
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = {random.choice(list(weights.keys())): random.randint(0, 10) for k 
              in range(random.randint(2, 20))}
        y = 1 if dotProduct(phi, weights) > 0 else -1
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        x_no_space = ''.join(x.split())
        word_count = {}
        for i in range(0, len(x_no_space) - n + 1):
            word_count[x_no_space[i:i+n]] = word_count.get(x_no_space[i:i+n], \
                0) + 1
        return word_count 
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    centers_prev = random.sample(examples, K)
    assignments = [0 for v in range(len(examples))]
    examples_self_dot = [dotProduct(v, v) for v in examples]
    centers = centers_prev
    def assign_points():
        nonlocal assignments   
        centers_self_dot = [dotProduct(c, c) for c in centers]
        for i in range(len(examples)):
            best_ass = -1
            best_distance = float('inf')
            for j in range(len(centers)):
                d = examples_self_dot[i] - 2 * dotProduct(examples[i], centers[j]) + \
                    centers_self_dot[j]
                if d < best_distance:
                    best_ass = j
                    best_distance = d
            assignments[i] = best_ass

    def assign_centers():
        nonlocal centers
        centers_unnormalized = [{} for c in range(K)]
        counts = [0 for c in range(K)]
        for i in range(len(assignments)):
            increment(centers_unnormalized[assignments[i]], 1, examples[i])
            counts[assignments[i]] += 1
        centers = [{} for c in range(K)]
        for i in range(K):
            increment(centers[i], 1 / counts[i], centers_unnormalized[i])

    def reconstruction_loss():
        loss = 0
        centers_self_dot = [dotProduct(c, c) for c in centers]
        for i in range(len(examples)):
            loss += examples_self_dot[i] - 2 * dotProduct(examples[i], \
                    centers[assignments[i]]) + centers_self_dot[assignments[i]]
        return loss

    for i in range(maxIters):
        assign_points()
        assign_centers()
        if centers_prev == centers:
            break
        else:
            centers_prev = centers
    return centers, assignments, reconstruction_loss()

    # END_YOUR_CODE
