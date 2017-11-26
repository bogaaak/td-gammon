import os
import tensorflow as tf
import numpy as np
from sortedcontainers import SortedSet
from model import Model
from structure_utils import *



flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('test', False, 'If true, test against a random strategy.')
flags.DEFINE_boolean('play', False, 'If true, play against a trained TD-Gammon strategy.')
flags.DEFINE_boolean('restore', False, 'If true, restore a checkpoint before training.')
flags.DEFINE_boolean('test_structures', False, 'If true, test and record structures in the environment.')

model_path = os.environ.get('MODEL_PATH', 'models/')
summary_path = os.environ.get('SUMMARY_PATH', 'summaries/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
#
# if not os.path.exists(model_path):
#     os.makedirs(model_path)
#
# if not os.path.exists(checkpoint_path):
#     os.makedirs(checkpoint_path)
#
# if not os.path.exists(summary_path):
#     os.makedirs(summary_path)

FLAGS.restore = True
FLAGS.test_structures = True

if __name__ == '__main__':
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default():
        with graph.as_default():
            model = Model(sess, model_path, summary_path, checkpoint_path, restore=FLAGS.restore)

            if FLAGS.test:
                model.test(episodes=1000)
            elif FLAGS.test_structures:
                print("bla")
                model.test_structures(episodes=20)
            elif FLAGS.play:
                model.play()
            else:
                model.train()




al = model.players[0].actions_list
len(al)
check_consistency_of_last_layer(al)

# mean_dom = [0.] * len(al)
mean_dom = dict.fromkeys(range(len(al)))
mean_dom_eq = dict.fromkeys(range(len(al)))
mean_cum_dom = dict.fromkeys(range(len(al)))
mean_cum_dom_eq = dict.fromkeys(range(len(al)))

number_of_actions = np.zeros(len(al), dtype=int)

for ix, a in enumerate(al):
    last_layer_weights = a.last_layer_weights
    feature_matrix = a.feature_matrix
    number_of_actions[ix] = a.n_actions
    dominated_columns, dominance_equivalent_columns, cum_dominated_columns, cum_dominance_equivalent_columns = \
        calc_dominance_brute_force(feature_matrix, last_layer_weights)
    mean_dom[ix] = np.mean(dominated_columns)
    mean_dom_eq[ix] = np.mean(dominance_equivalent_columns)
    mean_cum_dom[ix] = np.mean(cum_dominated_columns)
    mean_cum_dom_eq[ix] = np.mean(cum_dominance_equivalent_columns)

np.mean(np.array(list(mean_cum_dom.values())))
np.mean(np.array(list(mean_cum_dom_eq.values())))


## Random baseline!
mean_dom = dict.fromkeys(range(len(al)))
mean_dom_eq = dict.fromkeys(range(len(al)))
mean_cum_dom = dict.fromkeys(range(len(al)))
mean_cum_dom_eq = dict.fromkeys(range(len(al)))
for ix, na in enumerate(number_of_actions):
    last_layer_weights = np.vstack(np.random.randn(50))
    feature_matrix = np.random.randn(50, int(np.ceil(na/2)))
    dominated_columns, dominance_equivalent_columns, cum_dominated_columns, cum_dominance_equivalent_columns = \
        calc_dominance_brute_force(feature_matrix, last_layer_weights)
    mean_dom[ix] = np.mean(dominated_columns)
    mean_dom_eq[ix] = np.mean(dominance_equivalent_columns)
    mean_cum_dom[ix] = np.mean(cum_dominated_columns)
    mean_cum_dom_eq[ix] = np.mean(cum_dominance_equivalent_columns)


np.mean(np.array(list(mean_dom.values())))
np.mean(np.array(list(mean_dom_eq.values())))
np.mean(np.array(list(mean_cum_dom.values())))
np.mean(np.array(list(mean_cum_dom_eq.values())))



# al[367].n_actions
# al[367].feature_matrix
# al[367].actions


# mean_dom = dict.fromkeys(range(len(al)))
# for ix, a in enumerate(al):  # ix = 1; a = al[ix]
#     actions_set = set(frozenset(i) for i in a.actions)
#     a.actions = actions_set
#     last_layer_weights = a.last_layer_weights
#     feature_matrix = a.feature_matrix
#     dominated_columns = calc_dominance_brute_force(feature_matrix, last_layer_weights)
#     mean_dom[ix] = np.mean(dominated_columns)
#
#
#     al[384].actions[ix] = set(al[384].actions[ix])
#
# al[384].actions[1]
#
# last_layer_weights = al[0].last_layer_weights
# feature_matrix = al[0].feature_matrix
#
# calc_dominance_brute_force(feature_matrix, last_layer_weights)
#
#
#




#
# graph = tf.Graph()
# sess = tf.InteractiveSession(graph=graph)
# model = Model(sess, model_path, summary_path, checkpoint_path, restore=FLAGS.restore)
# tf.trainable_variables()
# sess.run([v for v in tf.trainable_variables() if v.name == "layer2/weight:0"][0])
# w = sess.run("layer2/weight:0")
# b = sess.run("layer2/bias:0")
#
#
# variables_names =[v.name for v in tf.trainable_variables()]
# values = sess.run(variables_names)
# for k,v in zip(variables_names, values):
#     print(k, v)
#
#
# ## Save Graph for tensorboard visualization
# file_writer = tf.summary.FileWriter("checkpoints/", graph)