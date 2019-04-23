# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #Obtenir la position actuelle de l'agent
        actualState = self.mdp.getStates()[2]
        #Obtenir la prochaine position de l'agent
        nextState = mdp.getTransitionStatesAndProbs(actualState, mdp.getPossibleActions(actualState)[0])
        #Obtenir la totalite des positions
        allStates = mdp.getStates()
        #Calculer a chaque iteration pour chaque position pour toutes les actions possibles la Qvalue
        for interation in range(iterations):
            initialValues = self.values.copy()
            for position in allStates:
                finalValue = None
                for action in self.mdp.getPossibleActions(position):
                    actualValue = self.computeQValueFromValues(position, action)
                    #Si la valeur finale obtenu est nulle ou inferieur a la valeur actuelle, elle prend la val actuelle
                    if finalValue is None or finalValue < actualValue:
                        finalValue = actualValue
                #Si la valeur final est nulle on lui assigne la valeur 0
                if finalValue is None:
                    finalValue = 0
                initialValues[position] = finalValue
            self.values = initialValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #On initialise la Q value a 0 par defaut
        qValue = 0
        #On recupere les potitions de transitions et les differentes probabilite de ces positions
        functionTransitionResult = self.mdp.getTransitionStatesAndProbs(state,action)
        #Pour chaque prochain etat et sa probabilite on cumule la Q value
        for nextState, stateProbability in functionTransitionResult:
            reward = self.mdp.getReward(state, action, nextState)
            valueDiscounted = self.discount * self.values[nextState]
            qValue += stateProbability * (reward + valueDiscounted)

        return qValue

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #Obtenir toutes ls actions possibles de l'agent et retourner la meilleur action a faire
        actions, maxValue, decision = self.mdp.getPossibleActions(state), -float('inf'), None
        for action in actions:
            actionValue = self.computeQValueFromValues(state, action)
            if actionValue > maxValue:
                maxValue = actionValue
                decision = action
        return decision

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
