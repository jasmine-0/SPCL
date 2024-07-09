import tensorflow as tf

class GradSimLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(GradSimLoss, self).__init__()
 
    def call(self, y_true, y_pred):
        loss = 1-tf.norm(y_true, ord=2)**2/(tf.norm(y_true, ord=2)**2+tf.norm(y_true-y_pred, ord=2)**2)
        # loss = 1-tf.norm(y_true, ord=2)**3/(tf.norm(y_true, ord=2)**3+2*tf.norm(y_true-y_pred, ord=2)**3)
        # loss = 1 - tf.norm(y_true, ord=2) ** 1 / (tf.norm(y_true, ord=2) ** 1 + tf.norm(y_true - y_pred, ord=2) ** 1)
        return loss

class ElcLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(ElcLoss, self).__init__()

    def call(self, y_true, y_pred):
        loss = 1-1/(1+tf.norm(y_true-y_pred, ord=2))
        return loss

class CosLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(CosLoss, self).__init__()

    def call(self, y_true, y_pred):
        loss = 1-tf.reduce_sum(tf.multiply(y_pred, y_true))/(tf.norm(y_true, ord=2)*tf.norm(y_pred, ord=2))
        return loss

class ElcNormLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(ElcNormLoss, self).__init__()

    def call(self, y_true, y_pred):
        loss = tf.norm(y_true-y_pred, ord=2)/(tf.norm(y_true, ord=2)+tf.norm(y_pred, ord=2))
        return loss
