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
        #till here
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        fns = [self.patches[i] for i in indices]

        x_batch = np.zeros((self.batch_size, self.input_size, self.input_size, self.num_channels), dtype=np.float32)
        y_sr_batch = np.zeros((self.batch_size, self.output_size, self.output_size, 8), dtype=np.float32)
        
        for i, (fn, state) in enumerate(fns):
            data = np.load(fn)

            # setup y_highres
            y_train_hr = data[:,:,4]
            y_train_hr[y_train_hr == 1] = 1#forest
            y_train_hr[y_train_hr == 2] = 1
            y_train_hr[y_train_hr == 3] = 1
            y_train_hr[y_train_hr == 4] = 1
            y_train_hr[y_train_hr == 5] = 1
            y_train_hr[y_train_hr == 6] = 2#shrublands
            y_train_hr[y_train_hr == 7] = 2
            y_train_hr[y_train_hr == 8] = np.random.choice(np.arange(1, 11), p=[0.15, 0.05, 0, 0.15, 0, 0, 0.15, 0, 0.5, 0])#savannas
            y_train_hr[y_train_hr == 9] = np.random.choice(np.arange(1, 11), p=[0.15, 0.05, 0, 0.15, 0, 0, 0.15, 0, 0.5, 0])
            y_train_hr[y_train_hr == 10] = 4#grassland
            y_train_hr[y_train_hr == 11] = 5#wetlands
            y_train_hr[y_train_hr == 12] = 6#croplands
            y_train_hr[y_train_hr == 13] = 7#builtup to impervious
            y_train_hr[y_train_hr == 14] = 6#cropland
            y_train_hr[y_train_hr == 15] = 10#ice
            y_train_hr[y_train_hr == 16] = 9#barren
            y_train_hr[y_train_hr == 17] = 10#water
            y_train_hr[y_train_hr > 10.] = 10

            try:
                y_train_hr = keras.utils.to_categorical(y_train_hr, 11)
                y_train_hr=np.delete(y_train_hr, [0,3,8], 2)
            except Exception as e:
                print("Here is the error")
                print(fn)
            y_sr_batch[i] = y_train_hr
            x_batch[i] = data[:,:,:4]
        return x_batch.copy(), y_sr_batch
    def on_epoch_end(self):#before beginning every epoch ye chaly ga so it shuffels
        self.indices = np.arange(len(self.patches))
        np.random.shuffle(self.indices)