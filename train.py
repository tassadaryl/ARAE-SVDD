import os
import logging
import tensorflow as tf
from itertools import islice

from models.autoencoder import Seq2Seq
from opts import configure_args
from build_vocab import Corpus
from utils import Params, Metrics, Checkpoints, set_logger, static_vars, set_seeds
from tensorboardX import SummaryWriter


def loss_func(targets, logits):
    # use SparseCat.Crossentropy since targets are not one-hot encoded
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    # not take zero (<pad>) target into account when computing the loss since sequence is padded
    mask = tf.math.logical_not(tf.math.equal(targets, 0))

    # accuracy
    idx = tf.cast(tf.math.argmax(logits, 2), dtype=tf.int32)
    match = tf.math.equal(idx, targets)
    logical_and = tf.math.logical_and(match, mask)
    #accuracy = tf.reduce_sum(tf.cast(logical_and, dtype=tf.int32)) / (targets.shape[0] * targets.shape[1]) * 100
    accuracy = tf.reduce_mean(tf.cast(logical_and, dtype=tf.float32)) * 100

    # crossentropy loss
    mask = tf.cast(mask, dtype=tf.int32)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss, accuracy


def train_autoencoder(models, optimizers, source, target, params):
    autoencoder = models
    ae_optim = optimizers
    with tf.GradientTape() as tape:
        logits = autoencoder(source, noise=True)
        ae_loss, accuracy = loss_func(target, logits)

    gradients = tape.gradient(ae_loss, autoencoder.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, params.clip)
    ae_optim.apply_gradients(zip(gradients, autoencoder.trainable_variables))

    return {'ae_loss': ae_loss.numpy(), 'acc': accuracy.numpy()}

def train_SVDD(autoencoder, optimizer, dataset, c, params, args):
    batch_epoch = params.batch_epoch
    step, batch = 0, 0
    for e in range(0, params.max_epoch):
        for batch, (source, target) in enumerate(dataset):
            
            with tf.GradientTape() as tape:
                output = autoencoder(source, encode_only=True, noise=False)
                dist_loss = tf.reduce_mean((output - c) ** 2)
            
            gradients = tape.gradient(dist_loss, autoencoder.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, params.clip)
            optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
            if step % 5 == 0:
                print("svdd loss: {}, step={}".format(dist_loss, step))
            step += 1


def train(models, optimizers, dataset, corpus, ckpts, params, args):
    epoch_num = params.epoch_num
    batch_epoch = params.batch_epoch
    autoencoder.noise_radius = params.noise_radius
    step = 0

    for e in range(epoch_num, params.max_epoch):
        for batch, (source, target) in islice(enumerate(dataset), batch_epoch, None):
            metrics = Metrics(
                epoch=e,
                max_epoch=params.max_epoch,
            )
            for p in range(params.epoch_ae):
                ae_metrics = train_autoencoder(models, optimizers, source, target, params)
                metrics.accum(ae_metrics)

            metrics['ae_loss'] /= params.epoch_ae
            metrics['acc'] /= params.epoch_ae

            batch_epoch += 1
            # anneal noise every 5 batch_epoch for now
            if batch_epoch % 5 == 0:
                autoencoder.noise_radius = autoencoder.noise_radius * 0.995
            if batch_epoch % params.print_every == 0:
                ckpts.save()
                logging.info('--- Epoch {}/{} Batch {} ---'.format(e + 1, metrics['max_epoch'], batch_epoch))
                logging.info('Loss {:.4f}'.format(float(metrics['ae_loss'])))

                params.batch_epoch = batch_epoch
                params.epoch_num = e
                params.noise_radius = autoencoder.noise_radius
                params.save(os.path.join(args.model_dir, 'params.json'))

                # Floydhub metrics
                print('{{"metric": "acc", "value": {}, "step": {}}}'.format(float(metrics['acc']), step))
                print('{{"metric": "ae_loss", "value": {}, "step": {}}}'.format(float(metrics['ae_loss']), step))


                step += 1
                tb_writer.add_scalar('train/acc', metrics['acc'], step)
                tb_writer.add_scalar('train/ae_loss', metrics['ae_loss'], step)

        batch_epoch = 0

def init_center_c(models, dataset, eps=0.1):
    return


if __name__ == '__main__':
    # Load the parameters from the experiment params.json file in model_dir
    args = configure_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    if not os.path.exists(os.path.join(args.model_dir, 'output')):
        os.makedirs(os.path.join(args.model_dir, 'output'))
    set_logger(os.path.join(args.model_dir, 'output/train.log'))
    set_seeds(args.seed)

    # Prepare dataset
    logging.info('Preparing dataset...')
    corpus = Corpus(args.data_dir, n_tokens=args.vocab_size)
    args.vocab_size = min(args.vocab_size, corpus.vocab_size)
    dataset = tf.data.Dataset.from_tensor_slices((corpus.train_source, corpus.train_target)).batch(params.batch_size, drop_remainder=True)

    # Models
    autoencoder = Seq2Seq(params, args)

    autoencoder.trainable = True

    # Optimizers
    ae_optim = tf.keras.optimizers.SGD(params.lr_ae)

    models = autoencoder #, discriminator, generator
    optimizers = ae_optim #, disc_optim, gen_optim

    tb_writer = SummaryWriter(logdir=args.model_dir)

    ckpts = Checkpoints(models, optimizers, os.path.join(args.model_dir, 'ckpts'))
    ckpts.restore()

    if ckpts.has_ckpts:
        logging.info("Restored from {}".format(ckpts.has_ckpts))
    else:
        logging.info("Initializing from scratch...")

    logging.info('Training...')
    train(models, optimizers, dataset, corpus, ckpts, params, args)

    ######################################################################
    ###         following are DeepSVDD part
    ######################################################################

    logging.info('Initializing c ...')
    batch_latent_mean = []
    for batch, (source, target) in enumerate(dataset):
        real_latent = autoencoder(source, encode_only=True, noise=False)
        batch_latent_mean.append(tf.reduce_mean(real_latent, axis = 0))
    
    c = tf.reduce_mean(tf.cast(batch_latent_mean, tf.float32), axis = 0)
    svdd_optim = tf.keras.optimizers.Adam(amsgrad = True)
    
    logging.info('Training SVDD...')

    train_SVDD(autoencoder, svdd_optim, dataset, c, params, args)














