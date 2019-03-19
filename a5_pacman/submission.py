from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
    def minimax(s, d, agent_index):
      if s.isWin() or s.isLose() or d == self.depth + 1:
        return self.evaluationFunction(s), None
      actions = s.getLegalActions(agent_index)
      next_agent_index = agent_index + 1 if agent_index < s.getNumAgents() - 1 else 0
      next_d = d if agent_index < s.getNumAgents() - 1 else d + 1
      values = [minimax(s.generateSuccessor(agent_index, action), next_d, next_agent_index)[0] for action in actions]
      if agent_index == 0:
        max_value = max(values) 
        max_index = values.index(max_value)
        return max_value, actions[max_index]
      else:
        min_value = min(values)
        min_index = values.index(min_value)
        return min_value, actions[min_index]

    value, action = minimax(gameState, 1, 0)
    return action
    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 49 lines of code, but don't worry if you deviate from this)

    def minimax_alpha_beta(s, d, agent_index, ancestor_min, ancestor_max):
      if s.isWin() or s.isLose() or d == self.depth + 1:
        return self.evaluationFunction(s), None
      actions = s.getLegalActions(agent_index)
      next_agent_index = agent_index + 1 if agent_index < s.getNumAgents() - 1 else 0
      next_d = d if agent_index < s.getNumAgents() - 1 else d + 1
      values = []
      best_value = 1e9 if agent_index >= 1 else -1e9
      best_action = None
      for action in actions:
        value, _ = minimax_alpha_beta(s.generateSuccessor(agent_index, action), next_d, next_agent_index, ancestor_min, ancestor_max)
        if agent_index >= 1:
          if value < best_value:
            best_value = value
            best_action = action
          ancestor_min = min(ancestor_min, best_value)
          if best_value <= ancestor_max:
            return best_value, best_action
        else:
          if value > best_value:
            best_value = value
            best_action = action
          ancestor_max = max(ancestor_max, best_value)
          if value >= ancestor_min:
            return best_value, best_action
      return best_value, best_action

    value, action = minimax_alpha_beta(gameState, 1, 0, 1e9, -1e9)
    print(value)
    return action
    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    def expectimax(s, d, agent_index):
      if s.isWin() or s.isLose() or d == self.depth + 1:
        return self.evaluationFunction(s), None
      actions = s.getLegalActions(agent_index)
      next_agent_index = agent_index + 1 if agent_index < s.getNumAgents() - 1 else 0
      next_d = d if agent_index < s.getNumAgents() - 1 else d + 1
      values = [expectimax(s.generateSuccessor(agent_index, action), next_d, next_agent_index)[0] for action in actions]
      if agent_index == 0:
        max_value = max(values) 
        max_index = values.index(max_value)
        return max_value, actions[max_index]
      else:
        random_index = random.randint(0, len(values) - 1)
        expected_value = values[random_index]
        action = actions[random_index]
        return expected_value, action

    value, action = expectimax(gameState, 1, 0)
    return action

    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: <write something here so we know what you did>
  """

  # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
  pacman_position = currentGameState.getPacmanPosition()
  ghost_positions = currentGameState.getGhostPositions()
  score = currentGameState.getScore()
  distance_to_ghosts = [manhattanDistance(pacman_position, ghost_position) for ghost_position in ghost_positions]
  min_ghost_distance = min(distance_to_ghosts)
  avg_ghost_distance = sum(distance_to_ghosts) / len(distance_to_ghosts)
  num_foods = currentGameState.getNumFood()
  foods = currentGameState.getFood()
  distance_to_foods = [manhattanDistance(pacman_position, (x, y)) for x in range(foods.width) for y in range(foods.height) if foods[x][y]]
  avg_food_distance = sum(distance_to_foods) / len(distance_to_foods) if len(distance_to_foods) > 0 else -10
  radius = 2
  count_food_nearby = sum(foods[x][y] for x in range(max(pacman_position[0] - radius, 0), min(pacman_position[0] + radius, foods.width)) for y in range(max(pacman_position[1] - radius, 0), min(pacman_position[1] + radius, foods.height)))
  # print("score: {}".format(score))
  # print("min ghost distance: {}".format(min_ghost_distance))
  # print("num foods: {}".format(num_foods))
  # print("")
  # return score + 10 * min_ghost_distance
  return 5 * score + 8 * avg_ghost_distance - 10 * avg_food_distance + 5 * count_food_nearby
  # return score
  # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
