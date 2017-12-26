import tensorflow as tf
import numpy as np

def min_loss(loss,optimizer,max_gradient_norm=None):
    if max_gradient_norm == None:
        train_op = optimizer.minimize(loss)
        return train_op
    else:
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params, colocate_gradients_with_ops=True)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        train_op = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        return train_op