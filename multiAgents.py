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
        score = successorGameState.getScore()

        # food evaluatie
        foodList = newFood.asList()
        if foodList:
            # dichtstbijzijnde food
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            minFoodDistance = min(foodDistances)
            if minFoodDistance > 0:
                score += 1.0 / minFoodDistance
                
        # ghost evaluatie
        for ghostState in newGhostStates:
            ghostDist = manhattanDistance(newPos, ghostState.getPosition())
            # als ghost te dichtbij is
            if ghostDist < 2:
                return -float('inf')  # zorg dat je niet doodgaat
            if ghostDist > 0:
                score -= 2.0 / ghostDist

        return score

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
        def minimax(state, agentIndex, depth):
            # check eerst of we in een eindstaat zijn
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
                
            numAgents = state.getNumAgents()
            
            # alle mogelijke acties
            legalActions = state.getLegalActions(agentIndex)
            
            # geen mogelijke acties
            if not legalActions:
                return self.evaluationFunction(state)
                
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth
            
            # initialiseer gebasseerd op agent (max voor pacman, min voor ghost)
            if agentIndex == 0:  # pacman
                value = float("-inf")
                for action in legalActions:
                    successorState = state.generateSuccessor(agentIndex, action)
                    value = max(value, minimax(successorState, nextAgent, nextDepth))
            else:  # ghost
                value = float("inf")
                for action in legalActions:
                    successorState = state.generateSuccessor(agentIndex, action)
                    value = min(value, minimax(successorState, nextAgent, nextDepth))
                    
            return value

        # beste actie voor pacman
        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return None
            
        # vind de actie met de hoogste waarde
        bestValue = float("-inf")
        bestAction = None
        
        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            value = minimax(successorState, 1, 0)
            if value > bestValue:
                bestValue = value
                bestAction = action
                
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #initialize vars - negative infinity as we want to maximize, and 0 as best move
        maxvar = -float('inf')
        mmaction = 0
        alphas = [-float('inf')]
        for _ in range(1, gameState.getNumAgents()):
          alphas.append(float('inf'))

        def findV(state, depth, alphas):
          #base case
          if (state.isWin() | state.isLose()) | (depth == self.depth*state.getNumAgents()):
            return self.evaluationFunction(state)

          # determine the current agent's index 
          agentIndex = depth % state.getNumAgents()
          isMax = agentIndex == 0
          return value(state, depth, agentIndex, alphas, isMax)


        def value(state, depth, agentIndex, alphas, isMax):
            alphas = alphas[:]  # create a copy of alpha values to avoid modifying the og list
            if isMax:
                v = -float('inf')  # Initialize v for maximizing player
                compf = max  # comparison function for maximizing player
                cvalue = min(alphas[1:])  # comparison value 
            else:
                v = float('inf')  # Initialize v for minimizing player
                compf = min  # comparison function for minimizing player
                cvalue = alphas[0]  # comparison value 

            # iterate through legal actions and calculate the value
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = compf(v, findV(successor, depth + 1, alphas))
                if isMax and v > cvalue:
                    return v
                elif not isMax and v < cvalue:
                    return v
                alphas[agentIndex] = compf(alphas[agentIndex], v)

            return v
      
        # check states and return the evaluation function value
        if gameState.isWin() | gameState.isLose():
            return self.evaluationFunction(gameState)

        # loop through actions for Pac-Man
        for action in gameState.getLegalActions(0):
            result = findV(gameState.generateSuccessor(0, action), 1, alphas) 
            if result > maxvar:  # update the best move and alpha value 
                mmaction = action
                maxvar = result
            alphas[0] = max(alphas[0], maxvar)

        return mmaction #return the best move
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
        def expectimax(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0: 
                return max(expectimax(state.generateSuccessor(agentIndex, action), nextAgent, nextDepth)
                           for action in legalActions)
            else:
                return sum(expectimax(state.generateSuccessor(agentIndex, action), nextAgent, nextDepth)
                           for action in legalActions) / len(legalActions)

        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return None

        bestAction = max(legalActions, key=lambda action: expectimax(gameState.generateSuccessor(0, action), 1, 0))
        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    A more comprehensive evaluation function that considers:
    1. Current game score
    2. Distance to closest food
    3. Number of remaining food
    4. Distance to ghosts
    5. Scared ghost states
    """
    # Extract key game state information
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Start with the current game score
    score = currentGameState.getScore()

    # Food evaluation
    foodList = food.asList()
    if foodList:
        # Incentivize moving closer to food
        closestFoodDist = min([manhattanDistance(pos, food) for food in foodList])
        score += 10.0 / (closestFoodDist + 1)  # avoid division by zero
        
        # Bonus for eating food, penalize for many remaining foods
        score += 100 / (len(foodList) + 1)

    # Ghost evaluation
    for i, ghostState in enumerate(ghostStates):
        ghostDist = manhattanDistance(pos, ghostState.getPosition())
        
        # If ghost is scared, get closer and try to eat it
        if scaredTimes[i] > 0:
            score += 50 / (ghostDist + 1)
        else:
            # Avoid ghosts, especially if they're close
            if ghostDist < 2:
                score -= 500  # High penalty for being near a non-scared ghost
            else:
                score += 10 / (ghostDist + 1)  # Small bonus for maintaining distance

    return score

# Abbreviation
better = betterEvaluationFunction
