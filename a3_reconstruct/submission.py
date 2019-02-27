import shell
import util
import wordsegUtil

############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        return (tuple([0 for i in range(len(self.query))]), -1)
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        return state[1] == len(self.query) - 1
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        results = []
        config, index = state
        for j in range(index + 1, len(self.query)):
            succ_config = list(config)
            succ_config[j] = 1
            results.append((j, (tuple(succ_config), j), 
                self.unigramCost(self.query[index + 1: j + 1])))
        return results
        # END_YOUR_CODE

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 10 lines of code, but don't worry if you deviate from this)
    actions = ucs.actions
    actions.insert(0, -1)
    words = []
    for i in range(len(actions) - 1):
        words.append(query[actions[i] + 1: actions[i + 1] + 1])
    return ' '.join(words) 
    # END_YOUR_CODE

############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return ()
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        return len(state) == len(self.queryWords)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 16 lines of code, but don't worry if you deviate from this)
        assert not self.isEnd(state)
        results = []
        prev_word = state[-1] if len(state) > 0 else wordsegUtil.SENTENCE_BEGIN
        if not self.possibleFills(self.queryWords[len(state)]):
            succ_state = list(state) 
            candidate = self.queryWords[len(state)]
            results.append((candidate, tuple(succ_state.append(candidate)), 
                self.bigramCost(prev_word, candidate)))
            return results
        for candidate in self.possibleFills(self.queryWords[len(state)]):
            succ_state = list(state) 
            succ_state.append(candidate)
            results.append((candidate, tuple(succ_state), self.bigramCost(
                prev_word, candidate)))
        return results
        # END_YOUR_CODE

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_CODE

############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        return ()
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        if not state:
            return False
        return state[-1][0] == len(self.query) - 1
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 23 lines of code, but don't worry if you deviate from this)
        results = []
        prev_word = state[-1][1] if len(state) > 0 else wordsegUtil.SENTENCE_BEGIN
        prev_index = state[-1][0] if len(state) > 0 else -1
        for next_index in range(prev_index + 1, len(self.query)):
            candidates = self.possibleFills(self.query[prev_index + 1: 
                next_index + 1])
            if not candidates:
                continue
            for candidate in candidates:
                succ_state = list(state)
                succ_state.append((next_index, candidate))
                results.append(((next_index, candidate), tuple(succ_state), 
                    self.bigramCost(prev_word, candidate)))
        return results
        # END_YOUR_CODE

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 11 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
    return ' '.join([action[1] for action in ucs.actions])
    # END_YOUR_CODE

############################################################

if __name__ == '__main__':
    shell.main()
