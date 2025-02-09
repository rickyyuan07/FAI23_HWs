# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        FOOD_WEIGHT = 10
        GHOST_WEIGHT = -8
        if successorGameState.isWin(): # food is empty
            return float("inf")
        nearestFood = min([manhattanDistance(newPos, food) for food in newFood.asList()])
        FoodScore = FOOD_WEIGHT / nearestFood
        nearestGhost = min([manhattanDistance(newPos,ghostState.configuration.pos) for ghostState in newGhostStates])
        GhostScore = GHOST_WEIGHT / nearestGhost if nearestGhost != 0 else 0
        if newScaredTimes[0] > 0: # only consider 1 ghost case
            GhostScore *= -1
        return successorGameState.getScore() + FoodScore + GhostScore

def scoreEvaluationFunction(currentGameState: GameState):
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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState: GameState, depth: int, agentIndex: int):
            if agentIndex == gameState.getNumAgents():
                agentIndex = 0
                depth += 1
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0: # pacman, max
                return max([minimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1) for action in gameState.getLegalActions(agentIndex)])
            else: # ghost, min
                return min([minimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1) for action in gameState.getLegalActions(agentIndex)])
        
        return max(gameState.getLegalActions(0), key=lambda x: minimax(gameState.generateSuccessor(0, x), 0, 1))
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState: GameState, depth: int, agentIndex: int, alpha: float, beta: float):
            if agentIndex == gameState.getNumAgents():
                agentIndex = 0
                depth += 1
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0: # pacman, max
                v = -float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    v = max(v, minimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta))
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                return v
            else: # ghost, min
                v = float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    v = min(v, minimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v
            
        alpha, beta, best_v = -float('inf'), float('inf'), -float('inf')
        best_action = None
        for action in gameState.getLegalActions(0):
            v = minimax(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            if v > best_v:
                best_v = v
                best_action = action
            alpha = max(alpha, v)
            
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(gameState: GameState, depth: int, agentIndex: int):
            if agentIndex == gameState.getNumAgents():
                agentIndex = 0
                depth += 1
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0: # pacman, max
                return max([expectimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1) for action in gameState.getLegalActions(agentIndex)])
            else: # ghost, expect
                return sum([expectimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1) for action in gameState.getLegalActions(agentIndex)]) / len(gameState.getLegalActions(agentIndex))
            
        return max(gameState.getLegalActions(0), key=lambda x: expectimax(gameState.generateSuccessor(0, x), 0, 1))
    

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Set the weight of food, capsule, ghost and scared ghost, which indicated the importance of each factor.
    If the game is win or lose, return the score directly.
    Food score is the reciprocal of the nearest food distance.
    Capsule score is the reciprocal of the nearest capsule distance.
    Ghost score is the sum of the reciprocal of the nearest ghost distance or the reciprocal of the nearest scared ghost distance. (based on the ghost state)
    The score is the sum of the food score, capsule score, ghost score and the current score.
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    FOOD_WEIGHT = 10
    CAPSULE_WEIGHT = 1
    GHOST_WEIGHT = -8
    SCARED_GHOST_WEIGHT = 100
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    if currentGameState.isWin():
        return currentGameState.getScore() # Note that if set float('inf'), the game will be stuck in the first step
    if currentGameState.isLose():
        return -float('inf')
    
    if len(Food.asList()) != 0:
        nearestFood = min([manhattanDistance(Pos, food) for food in Food.asList()])
        FoodScore = FOOD_WEIGHT / nearestFood
    else:
        FoodScore = FOOD_WEIGHT
    if len(capsules) != 0:
        nearestCapsule = min([manhattanDistance(Pos, capsule) for capsule in capsules])
        CapsuleScore = CAPSULE_WEIGHT / nearestCapsule
    else:
        CapsuleScore = CAPSULE_WEIGHT
    GhostScore = 0
    for ghost in GhostStates:
        dist = manhattanDistance(Pos, ghost.configuration.pos)
        if dist > 0:
            GhostScore += (GHOST_WEIGHT / dist) if ghost.scaredTimer == 0 else (SCARED_GHOST_WEIGHT / dist)

    return currentGameState.getScore() + FoodScore + GhostScore + CapsuleScore


# Abbreviation
better = betterEvaluationFunction
