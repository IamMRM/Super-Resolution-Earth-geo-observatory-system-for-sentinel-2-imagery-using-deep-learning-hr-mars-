from collections import defaultdict
import numpy as np
import keras.utils
import keras.backend as K

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, patches, batch_size, steps_per_epoch, input_size, output_size, num_channels):
        'Initialization'
        self.patches = patches
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        assert steps_per_epoch * batch_size < len(patches)

        self.input_size = input_size
        self.output_size = output_size

        self.num_channels = num_channels
        self.num_classes = 8

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def normalize(self, array):
        array_min, array_max = array.min(), array.max()
        return ((array - array_min) / (array_max - array_min))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        fns = [self.patches[i] for i in indices]
        x_batch = np.zeros((self.batch_size, self.input_size, self.input_size, self.num_channels), dtype=np.float32)
        y_hr_batch = np.zeros((self.batch_size, self.output_size, self.output_size, 8), dtype=np.float32)
        #y_sr_batch = np.zeros((self.batch_size, self.output_size, self.output_size, 11), dtype=np.float32)
        for i, (fn, state) in enumerate(fns):
            data = np.load(fn)
            #x_batch[i] = 2*(data[:, :, :4])/10000.0 - 1
            #x_batch[i] = data[:, :, :4]/10000.0
            redn = self.normalize(data[:, :, 0])
            greenn =self.normalize(data[:, :, 1])
            bluen =self.normalize(data[:, :, 2])
            infraredn=self.normalize(data[:, :, 3])
            x_batch[i] = np.dstack((redn, greenn, bluen,infraredn))

            y_train_nlcd = np.copy(data[:, :, 5])
            y_train_nlcd[y_train_nlcd == 1] = 0# forest
            y_train_nlcd[y_train_nlcd == 2] = 1
            y_train_nlcd[y_train_nlcd == 4] = 2
            y_train_nlcd[y_train_nlcd == 5] = 3
            y_train_nlcd[y_train_nlcd == 6] = 4
            y_train_nlcd[y_train_nlcd == 7] = 5
            y_train_nlcd[y_train_nlcd == 9] = 6
            y_train_nlcd[y_train_nlcd == 10] = 7
            y_train_nlcd = keras.utils.to_categorical(y_train_nlcd, 8)
            y_hr_batch[i] = y_train_nlcd
        return x_batch.copy(), {"outputs_hr": y_hr_batch, "outputs_sr": y_hr_batch}

    def on_epoch_end(self):#before beginning every epoch ye chaly ga so it shuffels
        self.indices = np.arange(len(self.patches))
        np.random.shuffle(self.indices)