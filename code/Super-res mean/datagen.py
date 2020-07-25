from collections import defaultdict
import numpy as np
import keras.utils
import keras.backend as K


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, patches, batch_size, steps_per_epoch, input_size, output_size, num_channels, do_color_aug=False,do_superres=True, superres_only_states=[]):
        'Initialization'
        self.patches = patches
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        assert steps_per_epoch * batch_size < len(patches)

        self.input_size = input_size
        self.output_size = output_size

        self.num_channels = num_channels
        self.num_classes = 8

        self.do_color_aug = do_color_aug

        self.do_superres = do_superres
        self.superres_only_states = superres_only_states
        self.on_epoch_end()

    def normalize(self, array):
        array_min, array_max = array.min(), array.max()
        return ((array - array_min) / (array_max - array_min))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        fns = [self.patches[i] for i in indices]
        x_batch = np.zeros((self.batch_size, self.input_size, self.input_size, self.num_channels), dtype=np.float32)
        y_hr_batch = np.zeros((self.batch_size, self.output_size, self.output_size, 8), dtype=np.float32)
        y_sr_batch = np.zeros((self.batch_size, self.output_size, self.output_size, 8), dtype=np.float32)
        for i, (fn, state) in enumerate(fns):
            data = np.load(fn)
            #x_batch[i] = 2*(data[:, :, :4])/10000.0 - 1
            #x_batch[i] = data[:, :, :4]/10000.0
            redn = self.normalize(data[:, :, 0])
            greenn =self.normalize(data[:, :, 1])
            bluen =self.normalize(data[:, :, 2])
            infraredn=self.normalize(data[:, :, 3])
            x_batch[i] = np.dstack((redn, greenn, bluen,infraredn))
            
            y_train_nlcd = np.copy(data[:, :, 4])
            y_train_nlcd[y_train_nlcd == 1] = 0# forest
            y_train_nlcd[y_train_nlcd == 2] = 0
            y_train_nlcd[y_train_nlcd == 3] = 0
            y_train_nlcd[y_train_nlcd == 4] = 0
            y_train_nlcd[y_train_nlcd == 5] = 0
            y_train_nlcd[y_train_nlcd == 6] = 1  # shrublands
            y_train_nlcd[y_train_nlcd == 7] = 1
            y_train_nlcd[y_train_nlcd == 8] = np.random.choice(np.arange(0, 8), p=[0.15, 0.05, 0.15, 0, 0, 0.15, 0.5, 0])  # savannas
            y_train_nlcd[y_train_nlcd == 9] = np.random.choice(np.arange(0, 8), p=[0.15, 0.05, 0.15, 0, 0, 0.15, 0.5, 0])
            y_train_nlcd[y_train_nlcd == 10] = 2  # grassland
            y_train_nlcd[y_train_nlcd == 11] = 3  # wetlands
            y_train_nlcd[y_train_nlcd == 12] = 4  # croplands
            y_train_nlcd[y_train_nlcd == 13] = 5  # builtup
            y_train_nlcd[y_train_nlcd == 14] = 4  # cropland
            y_train_nlcd[y_train_nlcd == 15] = 7  # ice
            y_train_nlcd[y_train_nlcd == 16] = 6  # barren
            y_train_nlcd[y_train_nlcd == 17] = 7  # water
            y_train_nlcd = keras.utils.to_categorical(y_train_nlcd, 8)
            y_hr_batch[i] = y_train_nlcd

        return x_batch.copy(), {"outputs_hr": y_hr_batch, "outputs_sr": y_hr_batch}

    def on_epoch_end(self):#before beginning every epoch ye chaly ga so it shuffels
        self.indices = np.arange(len(self.patches))
        np.random.shuffle(self.indices)