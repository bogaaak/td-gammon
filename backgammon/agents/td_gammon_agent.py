import numpy as np
import sys

def sigmoidNumpy(x):
    return 1 / (1 + np.exp(-x))

class TDAgent(object):
    counter = 0

    def __init__(self, player, model):
        self.player = player
        self.model = model
        self.name = 'TD-Gammon'

    def get_action(self, actions, game):
        """
        Return best action according to self.evaluationFunction,
        with no lookahead.
        """
        v_best = 0
        a_best = None

        for a in actions:
            ateList = game.take_action(a, self.player)
            features = game.extract_features(game.opponent(self.player))
            last_layer_features, last_layer_weights, last_layer_bias = self.model.get_last_layer(features)
            print(sigmoidNumpy(np.dot(last_layer_features, last_layer_weights) + last_layer_bias))
            v = self.model.get_output(features)
            print(v)
            TDAgent.counter += 1
            if TDAgent.counter > 10:
                sys.exit("sd")
            v = 1. - v if self.player == game.players[0] else v
            if v > v_best:
                v_best = v
                a_best = a
            game.undo_action(a, self.player, ateList)

        return a_best
