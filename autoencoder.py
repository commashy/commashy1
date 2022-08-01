import pickle as pk
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import Model, Input
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, Activation
from keras.layers import  Layer
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa
from src.equation_generation import *
from pathlib import Path
import gc

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
kerasBKED = os.environ["KERAS_BACKEND"] 
print(kerasBKED)

tf.compat.v1.disable_eager_execution()

path_to_training_data = Path.cwd().joinpath("reaction_data.pkl")
with open('reaction_data.pkl', 'rb') as f:
    data = pk.load(f)

dataset = data["training_data"]
tokenizer = data["tokenizer"]

number_of_equations = dataset.shape[0]
np.random.shuffle(dataset)

#current training method is unstable if batch size is not a fraction of the length of training data
training = dataset[:838860].astype(np.int32)
test = dataset[838860:1048575].astype(np.int32)

#setting up hyper parameters
batch_size = 20
epochs = 5000
max_length_of_equation = len(dataset[1])
latent_dimension = 350
intermediate_dimension = 500
epsilon_std = 0.1
kl_weight = 0.1
number_of_letters = len(tokenizer.word_index)
learning_rate = 1e-5
optimizer = Adam(learning_rate=learning_rate)

#Start model construction
input = Input(shape=(max_length_of_equation,))
embedded_layer = Embedding(number_of_letters, intermediate_dimension, input_length=max_length_of_equation)(input)
latent_vector = Bidirectional(LSTM(intermediate_dimension, return_sequences=False, recurrent_dropout=0.2), merge_mode='concat')(embedded_layer)
z_mean = Dense(latent_dimension)(latent_vector)
z_log_var = Dense(latent_dimension)(latent_vector)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dimension), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dimension,))([z_mean, z_log_var])
repeated_context = RepeatVector(max_length_of_equation)
decoder_latent_vector = LSTM(intermediate_dimension, return_sequences=True, recurrent_dropout=0.2)
decoder_mean = Dense(number_of_letters, activation='linear')#softmax is applied in the seq2seqloss by tf #TimeDistributed()
latent_vector_decoded = decoder_latent_vector(repeated_context(z))
input_decoded_mean = decoder_mean(latent_vector_decoded)

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
        self.target_weights = tf.constant(np.ones((batch_size, max_length_of_equation)), tf.float32)

    def vae_loss(self, x, x_decoded_mean):
        labels = tf.cast(x, tf.int32)
        xent_loss = K.sum(tfa.seq2seq.sequence_loss(x_decoded_mean, labels, 
                                                     weights=self.target_weights,
                                                     average_across_timesteps=False,
                                                     average_across_batch=False), axis=-1)#,
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        xent_loss = K.mean(xent_loss)
        kl_loss = K.mean(kl_loss)
        return K.mean(xent_loss + kl_weight * kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        print(x.shape, x_decoded_mean.shape)
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        return K.ones_like(x)

# placeholder loss
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

def kl_loss(x, x_decoded_mean):
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    kl_loss = kl_weight * kl_loss
    return kl_loss

loss_layer = CustomVariationalLayer()([input, input_decoded_mean])
vae = Model(input, [loss_layer])

vae.compile(optimizer=optimizer, loss=[zero_loss], metrics=[kl_loss])
vae.summary() 

def create_model_checkpoint(dir, model_name):
    filepath = dir + '/' + model_name + ".h5"
    directory = os.path.dirname(filepath)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    return ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)

del data
del dataset
gc.collect()

#======================= Model training ==============================#
checkpointer = create_model_checkpoint('models', 'agoras_checkpoints')

vae.fit(training, training,
      epochs=epochs,
      batch_size=batch_size,
      validation_data=(test, test), callbacks=[checkpointer])

print(K.eval(vae.optimizer.lr))
K.set_value(vae.optimizer.lr, learning_rate)

path = Path.cwd().joinpath("models")
vae.save_weights(path.joinpath("agoras_vae.h5"))
