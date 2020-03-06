from tensorflow.keras import layers
import tensorflow as tf


class MNIST_GAN:
    ''' Generative Adversarial Network to create MNIST digits '''
    def __init__(self, latent_dimension):
        # Create networks
        self.generator = self._create_generator(latent_dimension)
        self.discriminator = self._create_discriminator()
        # Create optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


    def _create_generator(self, latent_dimension):
        '''  Creates a generator that takes batchs of N gaussian noise values
            and produce a 28x28 matrix representing the output image 
            @param latent_dimension: The size of the latent dimension
            @return: The generator network
        '''
        model = tf.keras.Sequential()
        model.add(layers.Dense(7*7*256, use_bias=False,
                               input_shape=(latent_dimension,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)

        model.add(layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),
                                         padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model


    def _create_discriminator(self):
        ''' Creates a discriminator that takes batchs of 28x28 images and returns a batch of 0 or 1, 
            where 1 means the image is real and 0 means the image is fake.
            @return: The discriminator network
        '''
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[28, 28, 1]))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output):
        ''' Calculates the loss of the discriminator.
            This is the sum of cross entropy of all 
            real images(1) classified as fake(0) and
            fake images classified as real.
            @return the total discriminator loss
        '''
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        ''' Calculates the loss of the generator.
            This is the cross entropy of the desired result(1) and the classification if the generated images.
            @return the generator loss
        '''
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        return cross_entropy(tf.ones_like(fake_output), fake_output)
