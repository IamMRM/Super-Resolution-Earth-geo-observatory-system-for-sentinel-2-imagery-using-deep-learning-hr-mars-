import os
import time
from collections import defaultdict

import numpy as np

import keras


def humansize(nbytes):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

def schedule_decay(epoch, lr, decay=0.001):
    if epoch>=10:
        lr = lr * 1/(1 + decay * epoch)
    return lr

def schedule_stepped(epoch, lr, step_size=10):
    if epoch > 0:
        if epoch % step_size == 0:
            return lr / 10.
        else:
            return lr
    else:
        return lr

def find_key_by_str(keys, needle):
    for key in keys:
        if needle in key:
            return key
    raise ValueError("%s not found in keys" % (needle))

class LandcoverResults(keras.callbacks.Callback):

    def __init__(self, log_dir=None, time_budget=None, verbose=False, model=None):
        
        self.mb_log_keys = None
        self.epoch_log_keys = None

        self.verbose = verbose
        self.time_budget = time_budget
        self.log_dir = log_dir

        self.batch_num = 0
        self.epoch_num = 0

        self.model_inst = model

        if self.log_dir is not None:
            self.train_mb_fn = os.path.join(log_dir, "minibatch_history.txt")
            self.train_epoch_fn = os.path.join(log_dir, "epoch_history.txt")

    def on_train_begin(self, logs={}):
        self.train_start_time = time.time()
        
    def on_batch_begin(self, batch, logs={}):
        self.mb_start_time = float(time.time())

    def on_batch_end(self, batch, logs={}):
        t = time.time() - self.mb_start_time

        if self.mb_log_keys is None and self.log_dir is not None:
            self.mb_log_keys = [key for key in list(logs.keys()) if key!="batch" and key!="size"]
            f = open(self.train_mb_fn,"w")
            f.write("Batch Number,Time Elapsed")
            for key in self.mb_log_keys:
                f.write(",%s" % (key))
            f.write("\n")
            f.close()

        if self.log_dir is not None:
            f = open(self.train_mb_fn,"a")
            f.write("%d,%f" % (self.batch_num, t))
            for key in self.mb_log_keys:
                f.write(",%f" % (logs[key]))
            f.write("\n")
            f.close()

        self.batch_num += 1
        
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = float(time.time())
    
    def on_epoch_end(self, epoch, logs=None):
        t = time.time() - self.epoch_start_time
        total_time = time.time() - self.train_start_time

        if self.time_budget is not None:
            if total_time >= self.time_budget:
                try:
                    self.model.stop_training = True
                except Exception:
                    pass
                try:
                    self.model_inst.stop_training = True
                except Exception:
                    pass

        if self.epoch_log_keys is None and self.log_dir is not None:
            self.epoch_log_keys = [key for key in list(logs.keys()) if key!="epoch"]
            f = open(self.train_epoch_fn, "w")
            f.write("Epoch Number,Time Elapsed")
            for key in self.epoch_log_keys:
                f.write(",%s" % (key))
            f.write("\n")
            f.close()

        if self.log_dir is not None:
            f = open(self.train_epoch_fn,"a")
            f.write("%d,%f" % (self.epoch_num, t))
            for key in self.epoch_log_keys:
                f.write(",%f" % (logs[key]))
            f.write("\n")
            f.close()

        self.epoch_num += 1