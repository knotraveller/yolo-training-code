import logging
import sys
import os



def init_logging(log_file, level=logging.INFO):
    global LOG
    # if os.path.exists(log_file):
    #     os.remove(log_file)
    log_dir = './log'
    LOGGER = log_file.split('.')[0] + '.log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    LOG = logging.getLogger(LOGGER)

    LOG.setLevel(level)
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(os.path.join(log_dir, LOGGER))

    ff = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    sf = logging.Formatter('%(name)s - %(message)s')

    file_handler.setFormatter(ff)
    stream_handler.setFormatter(sf)
    
    LOG.addHandler(stream_handler)
    LOG.addHandler(file_handler)

    LOG.debug('Logger initialized')

    return LOG

def on_train_epoch_start(trainer):
    # global LOG
    # if trainer.epoch % 10 == 0:
    #     LOG.debug(f'Starting epoch {trainer.epoch+1}')
    pass

def on_fit_epoch_end(trainer):
    global LOG
    # if trainer.epoch % 10 == 0:
    LOG.debug(f'Finished epoch {trainer.epoch+1}' \
              f', train_loss: {trainer.loss:.4f}' \
              f', Current fitness: {trainer.fitness:.4f}' \
              f', Best fitness: {trainer.best_fitness:.4f}')
