import numpy as np
import sys

def sigmoidNumpy(x):
    return 1 / (1 + np.exp(-x))

class ActionData(object):
    def __init__(self, counter, n_actions, feature_matrix, last_layer_weights, last_layer_bias):
        self.counter = counter
        self.n_actions = n_actions
        self.feature_matrix = feature_matrix
        self.last_layer_weights = last_layer_weights
        self.last_layer_bias = last_layer_bias


class TDAgent(object):
    def __init__(self, player, model):
        self.player = player
        self.model = model
        self.name = 'TD-Gammon'
        self.layer_size_hidden = self.model.layer_size_hidden
        self.counter = 0
        self.actions_list = []

    def get_action(self, actions, game):
        """
        Return best action according to self.evaluationFunction,
        with no lookahead.
        """
        v_best = 0
        a_best = None
        print(actions)
        n_actions = len(actions)
        print(n_actions)
        if n_actions > 1000:
            print(n_actions)
        feature_matrix = np.zeros(shape=[self.layer_size_hidden, n_actions])
        last_layer_weights, last_layer_bias = self.model.get_last_layer_weights_and_bias()

        for ix, a in enumerate(actions):
            ateList = game.take_action(a, self.player)
            features = game.extract_features(game.opponent(self.player))
            last_layer_features = self.model.get_last_layer_features(features)
            feature_matrix[:, ix] = last_layer_features
            # print(last_layer_weights)
            # print(sigmoidNumpy(np.dot(last_layer_features, last_layer_weights) + last_layer_bias))
            v = self.model.get_output(features)
            # print(v)
            ## Exit for testing purposes
            # if TDAgent.counter > 10:
            #     sys.exit("sd")
            v = 1. - v if self.player == game.players[0] else v
            if v > v_best:
                v_best = v
                a_best = a
            game.undo_action(a, self.player, ateList)

        # Store action data
        action_data = ActionData(self.counter, n_actions, feature_matrix, last_layer_weights, last_layer_bias)
        self.actions_list.append(action_data)
        self.counter += 1
        return a_best

