# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qvalues = {}

        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #Definir le tuple qvalue
        qValueTuple = (state, action)
        null = 0.0
        #Si la position et l'action n'existe pas alors retourner null sinon retourner la qvalue de cette position-action
        if qValueTuple not in self.qvalues:
            return null
        else:
            return self.qvalues[qValueTuple]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #Recuperer toutes les positions et actions possible
        possibleActions = self.getLegalActions(state)
        value = None
        #Pour chaque action possible obtenir la qvalue
        for action in possibleActions:
            actualQValue = self.getQValue(state, action)
            #Si la qvalue est superieur a la value de l'action
            if actualQValue >= value:
                #La value recupere la valeur de l'action
                value = actualQValue
        #Si a la fin des iterations si value est different de null retourner sa valeur
        if value != None:
            return value
        #Sinon retourner zero
        else:
            return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #Recuperer la qvalue optimale et les actions possible de l'agent
        optimalQValue = self.computeValueFromQValues(state)
        optimalAction = None
        possibleActions = self.getLegalActions(state)
        #Si aucune action n'est possible alors retourner None
        if not possibleActions:
            return None
        #Initialiser une liste pour receuillir les actions optimale
        optimalActionList = []
        #Pour chaque action possible
        for action in possibleActions:
            #Si la qvalue de l'action est egale a la qvalue optimal trouve pour cette action, ajouter l'action a la liste
            if self.getQValue(state, action) == optimalQValue:
                optimalActionList.append(action)
        #Au cas ou plusieurs actions optimal sont trouvees choisir l'une des valeurs au hasard
        optimalAction = random.choice(optimalActionList)
        return optimalAction


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)

        if len(legalActions) == 0:
            return None

        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)

        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #Verifier si le tuple position-action est ou pas dans les, si il n'y est pas alors initialiser la valeur a 0.0
        qValueTuple = (state, action)
        if qValueTuple not in self.qvalues:
            self.qvalues[qValueTuple] = 0.0
        #Calculer la value de la position suivante et qvalue position actuelle
        nextStateValue = self.computeValueFromQValues(nextState)
        actualStateValue = self.qvalues[(state, action)]
        #Faire le calcule en appliquant la formule
        calculation = reward + (self.discount * nextStateValue) - actualStateValue
        #Mettre a jour la qvalue pour la position et l'action
        self.qvalues[(state, action)] = actualStateValue + (self.alpha * calculation)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #Extraire le vecteur des features a partir de la position et de l'action
        qValue = 0.0
        featureVector = self.featExtractor.getFeatures(state, action)
        #Pour chaque feature dans le vecteur, recuperer le poids de ce feature et mettre a jour la q value
        for feature in featureVector:
            weigth = self.weights[feature]
            qValue += weigth * featureVector[feature]
        #Retourner la qValue obtenu apres l'iteration
        return qValue
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #Calculer la difference avec la recompense, l'escompte, la qValue et la valeur du prochain etat
        difference = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        #Extraire le vecteur des features a partir de la position et de l'action
        featureVector = self.featExtractor.getFeatures(state, action)
        #Pour chaque feature dans le vecteur, calculer le poids a partir de la difference  et mettre a jour le poids du feature
        for feature in featureVector:
            weight = self.alpha * difference * featureVector[feature]
            self.weights[feature] += weight
        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
