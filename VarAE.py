import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.io

class Sampling(layers.Layer):
  
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

nFeatures = 5 #+1 to account for droplet diameter size
latent_dim = nFeatures
initNNodes = 512

alpha = 0.3
encoder_inputs = keras.Input(shape=nFeatures)
nNodes = initNNodes

enc = keras.layers.Dense(initNNodes)(encoder_inputs)
enc = keras.layers.LeakyReLU(alpha)(enc)
enc = keras.layers.Dropout(0.5)(enc)
enc = keras.layers.BatchNormalization()(enc)
enc = keras.layers.Dense(initNNodes)(enc)
enc = keras.layers.LeakyReLU(alpha)(enc)
enc = keras.layers.Dropout(0.5)(enc)
enc = keras.layers.BatchNormalization()(enc)

z_mean = layers.Dense(latent_dim, name="z_mean")(enc)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(enc)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
n = latent_dim

dec = keras.layers.Dense(initNNodes)(latent_inputs)
dec = keras.layers.LeakyReLU(alpha)(dec)
dec = keras.layers.Dropout(0.5)(dec)
dec = keras.layers.BatchNormalization()(dec)
dec = keras.layers.Dense(initNNodes)(dec)
dec = keras.layers.LeakyReLU(alpha)(dec)
dec = keras.layers.Dropout(0.5)(dec)
dec = keras.layers.BatchNormalization()(dec)

decoder_outputs = keras.layers.Dense(nFeatures, activation='sigmoid')(dec)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            #Mean squared Error as loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction)
                )
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
