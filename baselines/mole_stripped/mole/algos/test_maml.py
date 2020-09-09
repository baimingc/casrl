import numpy as np
import tensorflow as tf
import random
import pickle
import csv
from tensorflow.python.platform import flags
import time

FLAGS = flags.FLAGS

# calculated for omniglot
NUM_TEST_POINTS = 600


def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []

    for _ in range(NUM_TEST_POINTS):
        if 'generate' not in dir(data_generator):
            feed_dict = {model.meta_lr: 0.0}
        else:
            batch_x, batch_y = data_generator.generate(train=False)

        inputa = batch_x[:, :FLAGS.update_batch_size, :]
        inputb = batch_x[:, FLAGS.update_batch_size:, :]
        labela = batch_y[:, :FLAGS.update_batch_size, :]
        labelb = batch_y[:, FLAGS.update_batch_size:, :]

        feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb,
                       model.meta_lr: 0.0}

        result = sess.run([model.total_loss1] + model.total_losses2, feed_dict)
        metaval_accuracies.append(result)

        metaval_accuracies = np.array(metaval_accuracies)
        means = np.mean(metaval_accuracies, axis=0)
        stds = np.std(metaval_accuracies, axis=0)
        ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)

        print('Mean validation accuracy/loss, stddev, and confidence intervals')
        print((means, stds, ci95))

        out_filename = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' \
                     + str(FLAGS.update_lr) + '.csv'
        out_pkl = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + \
                str(FLAGS.update_lr) + '.pkl'
        # saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model_test')
        with open(out_pkl, 'wb') as f:
            pickle.dump({'mses': metaval_accuracies}, f)
        with open(out_filename, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['update' + str(i) for i in range(len(means))])
            writer.writerow(means)
            writer.writerow(stds)
            writer.writerow(ci95)