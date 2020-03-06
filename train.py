import os
import time
from progress.bar import Bar
import matplotlib.pyplot as plt
import tensorflow as tf
from gan import MNIST_GAN


''' Load and Preporcess Data '''
# Load dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
# Reshape data and encode data
train_images = train_images.reshape(
    train_images.shape[0], 28, 28, 1).astype('float32')
# Normalize data to -1, 1
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = len(train_images)
BATCH_SIZE = 256

# Batch and Shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(
    train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


''' Create Models '''
mnist_gan = MNIST_GAN(100)


''' Check Generator and Descriminator are working '''
# # Test gen working
# noise = tf.random.normal([1, 100])
# generated_image = mnist_gan.generator(noise, training=False)
# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()

# # Test desc working
# decision = mnist_gan.discriminator(generated_image)
# print(decision)


''' Create Save Checkpoints '''
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "checkpoint")
checkpoint = tf.train.Checkpoint(generator_optimizer=mnist_gan.generator_optimizer,
                                 discriminator_optimizer=mnist_gan.discriminator_optimizer,
                                 generator=mnist_gan.generator,
                                 discriminator=mnist_gan.discriminator)

# Load models from last checkpoint
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
num_prev_epochs = len([name for name in os.listdir('training_images')])

''' Define Training Loop '''
EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16
training_image_inputs = tf.random.normal(
    [num_examples_to_generate, noise_dim], seed=0)


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = mnist_gan.generator(noise, training=True)

        real_output = mnist_gan.discriminator(images, training=True)
        fake_output = mnist_gan.discriminator(generated_images, training=True)

        generator_loss = mnist_gan.generator_loss(fake_output)
        descriminator_loss = mnist_gan.discriminator_loss(
            real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        generator_loss, mnist_gan.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        descriminator_loss, mnist_gan.discriminator.trainable_variables)

    mnist_gan.generator_optimizer.apply_gradients(
        zip(gradients_of_generator, mnist_gan.generator.trainable_variables))
    mnist_gan.discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, mnist_gan.discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(num_prev_epochs, epochs):
        start_time = time.time()

        num_batches = int(BUFFER_SIZE / BATCH_SIZE)
        with Bar('Epoch {}'.format(epoch), max=num_batches) as bar: 
            for image_batch in dataset:
                train_step(image_batch)
                bar.next()

        # Produce images for training GIF
        generate_and_save_images(
            mnist_gan.generator, epoch + 1, training_image_inputs)

        # Save the model every N epochs
        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time()-start_time))

    # Generate after the final epoch
    generate_and_save_images(
        mnist_gan.generator, epochs, training_image_inputs)


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('training_images/image_at_epoch_{:04d}.png'.format(epoch))


# TODO make main
train(train_dataset, EPOCHS)
