import os
import tensorflow as tf
import matplotlib.pyplot as plt
from gan import MNIST_GAN


mnist_gan = MNIST_GAN(100)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "checkpoint")
checkpoint = tf.train.Checkpoint(generator_optimizer=mnist_gan.generator_optimizer,
                                 discriminator_optimizer=mnist_gan.discriminator_optimizer,
                                 generator=mnist_gan.generator,
                                 discriminator=mnist_gan.discriminator)

# Load models from last checkpoint
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
num_prev_epochs = len([name for name in os.listdir('training_images')])

noise = tf.random.normal([1, 100])
generated_image = mnist_gan.generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()
