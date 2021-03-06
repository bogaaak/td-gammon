import os
import tensorflow as tf

from model import Model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('test', False, 'If true, test against a random strategy.')
flags.DEFINE_boolean('play', False, 'If true, play against a trained TD-Gammon strategy.')
flags.DEFINE_boolean('restore', False, 'If true, restore a checkpoint before training.')

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

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
model = Model(sess, model_path, summary_path, checkpoint_path, restore=FLAGS.restore)
file_writer = tf.summary.FileWriter("checkpoints/", graph)