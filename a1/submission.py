import collections
import math
from collections import defaultdict
import numpy as np
############################################################
# Problem 3a

def findAlphabeticallyLastWord(text):
    """
    Given a string |text|, return the word in |text| that comes last
    alphabetically (that is, the word that would appear last in a dictionary).
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() and list comprehensions handy here.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return max(text.split())
    # END_YOUR_CODE

############################################################
# Problem 3b

def euclideanDistance(loc1, loc2):
    """
    Return the Euclidean distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[2])**2)
    # END_YOUR_CODE

############################################################
# Problem 3c
def appendPair(pairs, output):
    new_output = []
    for i, s in enumerate(output):
        for j, pair in enumerate(pairs):
            if pairs[s[-1]][1] ==  pair[0]:
                new_s = s.copy()
                new_s.append(j)
                new_output.append(new_s)
    return new_output

def convert2String(pairs, output):
    string_output = []
    for o in output:
        sentence_list = [pairs[pair_index][0] for pair_index in o]
        sentence_list.append(pairs[o[-1]][1])
        string_output.append(' '.join(sentence_list))
    return string_output
    
def mutateSentences(sentence):
    """
    Given a sentence (sequence of words), return a list of all "similar"
    sentences.
    We define a sentence to be similar to the original sentence if
      - it as the same number of words, and
      - each pair of adjacent words in the new sentence also occurs in the original sentence
        (the words within each pair should appear in the same order in the output sentence
         as they did in the orignal sentence.)
    Notes:
      - The order of the sentences you output doesn't matter.
      - You must not output duplicates.
      - Your generated sentence can use a word in the original sentence more than
        once.
    Example:
      - Input: 'the cat and the mouse'
      - Output: ['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']
                (reordered versions of this list are allowed)
    """
    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
    words = sentence.split()
    n = len(words)
    pairs = list(set([(words[i], words[i + 1]) for i in range(0, n - 1)]))
    output = [[i] for i in range(0, len(pairs))]
    for i in range(1, n - 1):
        output = appendPair(pairs, output)     
    return convert2String(pairs, output)
    # END_YOUR_CODE

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collections.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    product_list = [v1[i] * v2[i] for i in v1.keys() if i in v2.keys()]
    return sum(product_list)
    # END_YOUR_CODE

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for i in v2.keys():
        v1[i] += scale * v2[i]
    # END_YOUR_CODE

############################################################
# Problem 3f

def findSingletonWords(text):
    """
    Splits the string |text| by whitespace and returns the set of words that
    occur exactly once.
    You might find it useful to use collections.defaultdict(int).
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    count = defaultdict(int)
    for word in text.split():
        count[word] += 1
    return set([word for word, c in count.items() if c == 1])
    # END_YOUR_CODE

############################################################
# Problem 3g

def computeLongestPalindromeLength(text):
    """
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.
    """
    # BEGIN_YOUR_CODE (our solution is 19 lines of code, but don't worry if you deviate from this)
    max_count = [[0 for i in range(0, len(text))] for j in range(0, len(text))]
    max_index = [[(None, None) for i in range(0, len(text))] for j in range(0, len(text))]
    for i in range(0, len(text)):
        max_count[i][i] = 1
        # first is None if there is no substring inside i, j
        # second is True if we choose the character at i, j as a character of longest sequence
        max_index[i][i] = (None, True) 
        for j in range(i - 1, -1, -1):
            if text[j] == text[i]:
                if i == j + 1:
                    max_count[j][i] = 2
                    max_index[j][i] = (None, True)
                else:
                    max_count[j][i] = max_count[j + 1][i - 1] + 2
                    max_index[j][i] = ((j + 1, i - 1), True)
            else:
                max_count[j][i] = max(max_count[j + 1][i], max_count[j][i - 1])
                if max_count[j][i] == max_count[j + 1][i]:
                    max_index[j][i] = ((j + 1, i), False)
                else:
                    max_index[j][i] = ((j, i - 1), False)
    max_length = max_count[0][len(text) - 1]
    longest_s_first_half = ''
    longest_s_second_half = ''
    j, i = 0, len(text) - 1
    while True:
        pair, max_at = max_index[j][i]
        if max_at:
            if j == i:
                longest_s_first_half += text[j]
            else:
                longest_s_first_half += text[j]
                longest_s_second_half = text[i] + longest_s_second_half
        if pair is not None:
            j, i = pair
        else:
            break
    longest_s = longest_s_first_half + longest_s_second_half
    return max_length, longest_s
            
    # END_YOUR_CODE
