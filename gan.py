from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import numpy as np

class VanillaGan(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 28*28])
        self.z = tf.placeholder(tf.float32, shape=[None, 128])
        self.g_var = None
        self.d_var = None
        self.g = self.generator(self.z)
        self.d_x = self.discriminator(self.x)
        self.d_g = self.discriminator(self.g)
        self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_x, labels=tf.ones_like(self.d_x))) +\
                 tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_g, labels=tf.zeros_like(self.d_g)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_g, labels=tf.ones_like(self.d_g)))
        self.d_opt = tf.train.AdamOptimizer().minimize(self.d_loss, var_list=self.d_var)
        self.g_opt = tf.train.AdamOptimizer().minimize(self.g_loss, var_list=self.g_var)


    def generator(self, z):
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            g_w1 = tf.get_variable("g_w1", [128, 256])
            g_w2 = tf.get_variable("g_w2", [256, 28*28])
            g_b1 = tf.get_variable("g_b1", [256])
            g_b2 = tf.get_variable("g_b2", [28*28])
            self.g_var = [g_w1, g_w2, g_b1, g_b2]
            h = tf.nn.relu(tf.matmul(z, g_w1) + g_b1)
            h = tf.matmul(h, g_w2) + g_b2
            out = tf.nn.sigmoid(h)
        return out


    def discriminator(self, x):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            d_w1 = tf.get_variable("d_w1", [28*28, 256])
            d_w2 = tf.get_variable("d_w2", [256, 1])
            d_b1 = tf.get_variable("d_b1", [256])
            d_b2 = tf.get_variable("d_b2", [1])
            self.d_var = [d_w1, d_w2, d_b1, d_b2]
            h = tf.nn.relu(tf.matmul(x, d_w1) + d_b1)
            logits = tf.matmul(h, d_w2) + d_b2
        return logits

def get_noise(batch, dim):
    return np.random.uniform(-1., 1., size=[batch, dim])

def train():
    mnist = input_data.read_data_sets("./MNIST/", one_hot=True)
    sess = tf.Session()

    gan = VanillaGan()

    d_loss_summary = tf.summary.scalar('d_loss', gan.d_loss)
    g_loss_summary = tf.summary.scalar('g_loss', gan.g_loss)
    train_writer = tf.summary.FileWriter("./summary/", sess.graph)
    sess.run(tf.global_variables_initializer())

    step = 0
    for epoch in range(120):
        num_batch = mnist.train.num_examples//128
        for batch in range(num_batch):
            batch_x, _ = mnist.train.next_batch(128)
            _, d_loss, d_summary = sess.run([gan.d_opt, gan.d_loss, d_loss_summary], feed_dict={gan.x: batch_x, gan.z: get_noise(128, 128)})
            _, g_loss, g_summary = sess.run([gan.g_opt, gan.g_loss, g_loss_summary], feed_dict={gan.z: get_noise(128, 128)})
            train_writer.add_summary(d_summary, step)
            train_writer.add_summary(g_summary, step)
            step += 1
            if step % 200 == 0:
                noise = get_noise(25, 128)
                samples = sess.run(gan.g, feed_dict={gan.z: noise})
                fig = plt.figure(figsize=(2.5, 2.5))
                gs = gridspec.GridSpec(5,5)
                gs.update(wspace=0.05, hspace=0.05)
                for i, sample in enumerate(samples):
                    ax = plt.subplot(gs[i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    plt.imshow(np.reshape(sample, (28,28)), cmap='gist_yarg')

                plt.savefig('./output/{}.png'.format(str(step/200).zfill(3)), bbox_inches='tight')
                plt.close(fig)
                print("[step #{}] g_loss : {:.4}\td_loss : {:.4}".format(step, g_loss, d_loss))


def test():
    #TODO
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    else:
        raise ValueError()


if __name__ == '__main__':
    main()
