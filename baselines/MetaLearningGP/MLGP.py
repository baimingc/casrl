import sys
import os

import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

import build_utils
import models


class Config_Parser(object):
    def __init__(self, mlgp_config):
        self.type = mlgp_config['optimizer']
        self.state_dim = mlgp_config['state_dim']
        self.action_dim = mlgp_config['action_dim']
        self.num_inducing = mlgp_config['num_inducing']
        self.num_tasks = mlgp_config['num_tasks']
        self.dim_in = mlgp_config['dim_in']
        self.dim_out = mlgp_config['dim_out']
        self.dim_latent = mlgp_config['dim_latent']


class MLGP(object):
    def __init__(self, mlgp_config):
        # initiate the model
        args = Config_Parser(mlgp_config=mlgp_config)
        model = models.MLGP(args.dim_in, args.dim_out, args.dim_latent, args.num_latent, args.num_inducing, args.multi_output)
        #model = models.TFSVGP(dim_in, dim_out, num_inducing, multi_output=multi_output)

    @staticmethod
    def data_process(data, discret=False):
        s = data[1]
        if discret:
            a = np.array([data[2]])
        else:
            a = data[2]
        s_n = data[3]
        data_point = np.concatenate((s, a, s_n), axis=0)[None]
        label = data[0]
        return data_point, label

    def fit(self, data, inference='SVI', discret=False):
        return None

    def predict(self, s, a):
        return ds



@gin.configurable
def eval_model(model, dataset, num_train_tasks, eval_dir, sample=False):
    num_tasks = dataset.num_tasks
    X_pred = dataset.X
    Y_pred = dataset.Y
    p_pred = np.int32(np.arange(num_tasks))[:, None, None]
    p_pred = np.tile(p_pred, [1, X_pred.shape[1], 1])

    inputs = {
        "X": X_pred.reshape(-1, X_pred.shape[2]),
        "p": p_pred.reshape(-1, 1)
    }

    Y_mu, Y_var = model.predict(inputs, sample=sample)
    Y_mu = Y_mu.numpy().reshape(num_tasks, X_pred.shape[1], -1)
    Y_var = Y_var.numpy().reshape(num_tasks, X_pred.shape[1], -1)

    plot_utils.plot_predictions(X_pred, Y_mu, Y_var, Y_pred, num_train_tasks, eval_dir)
    

@gin.configurable
def meta_inference(
    model, dataset, 
    num_train_tasks, 
    inf_dir, 
    num_inf_obs=10, 
    learning_rate=3e-4, 
    num_epochs=100, 
    batch_size=50, 
    reshuffle=True, 
    shuffle_buffer=10000, 
    restore_weights=True, 
    save_every=1000
):

    num_eval_tasks = dataset.num_tasks - num_train_tasks

    checkpoint_path = inf_dir + 'cp-{epoch:04d}.ckpt'
    cp_callback = tfk.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=save_every)
    
    optimizer = tfk.optimizers.Adam(learning_rate)
    num_train_obs = num_train_tasks*dataset.X.shape[1]
    num_eval_obs = num_eval_tasks*num_inf_obs
    num_obs = num_train_obs + num_eval_obs
    num_tasks = num_train_tasks + num_eval_tasks

    loss_fn = lambda Y, F: model.objective(Y, F, num_obs, num_tasks)
    # disable the training of GP
    model.GP.trainable = False
    model.compile(optimizer, loss=loss_fn)

    inf_ds = dataset.create_tf_dataset(batch_size, reshuffle, shuffle_buffer, num_obs=num_inf_obs)
    
    callbacks = [cp_callback,
        tfk.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=1e-10,
            mode='auto',
            patience=10000,
            verbose=1,
            restore_best_weights=restore_weights)
        ]
    
    model.fit(inf_ds, epochs=num_epochs, callbacks=callbacks)


@gin.configurable
def train(
    model, dataset, num_train_tasks, train_dir,
    learning_rate=3e-4,
    num_epochs=100,
    batch_size=50,
    reshuffle=True,
    shuffle_buffer=10000,
    restore_weights=True,
    save_every=1000,
    min_delta=1e-3,
    patience=5
):
    checkpoint_path = train_dir + 'cp-{epoch:04d}.ckpt'
    cp_callback = tfk.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, save_freq=save_every)

    optimizer = tfk.optimizers.Adam(learning_rate)
    num_train_obs = num_train_tasks*dataset.X.shape[1]
    loss_fn = lambda Y, F: model.objective(Y, F, num_train_obs, num_train_tasks)
    model.compile(optimizer, loss=loss_fn)
    
    model.save_weights(checkpoint_path.format(epoch=0))

    train_ids = np.int32(np.arange(num_train_tasks))
    train_ds = dataset.create_tf_dataset(batch_size, reshuffle, shuffle_buffer, p=train_ids)
    
    callbacks = [cp_callback, tfk.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=min_delta,
        mode='auto',
        patience=patience,
        verbose=1,
        restore_best_weights=restore_weights)
    ]
    
    model.fit(train_ds, epochs=num_epochs, callbacks=callbacks)


@gin.configurable
def main(exp_dir, model_name, dataset, num_train_tasks, seed, train_flag=True, inf_flag=True, eval_flag=True):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_dir = os.path.join(exp_dir, 'train/')
    check_create_dir(train_dir)
    inf_dir = os.path.join(exp_dir, 'inf/')
    check_create_dir(inf_dir)
    eval_dir = os.path.join(exp_dir, 'eval/')
    check_create_dir(eval_dir)

    if dataset == 'toy':
        dataset = build_utils.create_toy_dataset()
    else:
        raise NotImplementedError()

    model = build_utils.create_model(model_name, dataset)

    if train_flag:
        train(model, dataset, num_train_tasks, train_dir)
    if inf_flag:
        latest = tf.train.latest_checkpoint(train_dir)
        model.load_weights(latest)
        meta_inference(model, dataset, num_train_tasks, inf_dir)
    if eval_flag:
        latest = tf.train.latest_checkpoint(inf_dir)
        model.load_weights(latest)
        eval_model(model, dataset, num_train_tasks, eval_dir)


if __name__ == '__main__':
    root_dir = sys.argv[1]
    model_name = sys.argv[2]
    dataset = sys.argv[3]
    num_train_tasks = sys.argv[4]
    seed = sys.argv[5]


    gin.parse_config_file('./config/base.gin')
    main(exp_dir, model_name, dataset, int(num_train_tasks), int(seed))
