import tensorflow as tf
from tensorflow.keras import Model, layers, models, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, \
    BatchNormalization, Activation, Dropout,GlobalAvgPool2D, Dense, ReLU, ZeroPadding2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow_addons.layers import WeightNormalization

class MLPModel(Model):
    def __init__(self, class_number, method):
        super(MLPModel, self).__init__()
        self.method = method
        self.d1         = Dense(128, activation='relu')
        self.d2         = Dense(64, activation='relu')
        if self.method == 'mdmtr':
            self.d3         = Dense(class_number, use_bias=False)
        else:
            self.d3         = Dense(class_number)

    def masked_softmax(self, scores, mask):
        scores = scores - tf.tile(tf.reduce_max(scores, axis=(1,), keepdims=True), [1, tf.shape(scores)[1]])
        exp_scores = tf.exp(scores)
        exp_scores *= mask
        exp_sum_scores = tf.reduce_sum(exp_scores, axis=1, keepdims=True)
        return exp_scores / (tf.tile(exp_sum_scores, [1, tf.shape(exp_scores)[1]])+1e-7) 

    def call(self, x, mask):
        # x = self.d1(x)
        x = self.d1(x)
        x = self.d2(x)
        if self.method == 'mdmtr':
            x = tf.nn.l2_normalize(x, axis=-1)
            x = self.d3(x)
        else:
            x = self.d3(x)
            x = self.masked_softmax(x, mask)
        return x


class BaseModel(Model):
    def __init__(self, class_number, method):
        super(BaseModel, self).__init__()
        self.method = method
        self.conv1      = Conv2D(32, 3, activation='relu')
        self.maxpool1   = MaxPool2D((2,2))
        self.conv2      = Conv2D(64, 3, activation='relu')
        self.maxpool2   = MaxPool2D((2,2))
        self.conv3      = Conv2D(64, 3, activation='relu')
        self.flatten    = Flatten()
        self.d1         = Dense(64, activation='relu')
        if self.method == 'mdmtr':
            self.d2         = Dense(class_number, use_bias=False)
        else:
            self.d2         = Dense(class_number)

    def masked_softmax(self, scores, mask):
        scores = scores - tf.tile(tf.reduce_max(scores, axis=(1,), keepdims=True), [1, tf.shape(scores)[1]])
        exp_scores = tf.exp(scores)
        exp_scores *= mask
        exp_sum_scores = tf.reduce_sum(exp_scores, axis=1, keepdims=True)
        return exp_scores / (tf.tile(exp_sum_scores, [1, tf.shape(exp_scores)[1]])+1e-7) 

    def call(self, x, mask, with_softmax=True):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.d1(x)
        if self.method == 'mdmtr':
            x = tf.nn.l2_normalize(x, axis=-1)
            x = self.d2(x)
        else:
            x = self.d2(x)
            x = self.masked_softmax(x, mask)
        return x


# for 18 or 34 layers
class Basic_Block(Model):
    ''' basic block constructing the layers for resNet18 and resNet34
    '''
    def __init__(self, filters, block_name, downsample=False, stride=1):
        self.expasion = 1
        super(Basic_Block, self).__init__()
        conv_name = 'res' + block_name + '_branch'
        bn_name = 'bn' + block_name + '_branch'

        self.downsample = downsample

        self.conv2a = Conv2D(filters=filters,
                                          kernel_size=3,
                                          padding='same',
                                          strides=stride,
                                          kernel_initializer='he_normal',
                                          )
        self.bn2a = BatchNormalization(axis=-1,fused=False)

        self.conv2b = Conv2D(filters=filters,
                                          kernel_size=3,
                                          padding='same',
                                          kernel_initializer='he_normal'
                                          )
        self.bn2b = BatchNormalization(axis=-1,fused=False)

        self.relu = ReLU()

        if self.downsample:
            self.conv_shortcut = Conv2D(filters=filters,
                                                     kernel_size=1,
                                                     strides=stride,
                                                     kernel_initializer='he_normal',
                                                     )
            self.bn_shortcut = BatchNormalization(axis=-1,fused=False)

    def call(self, inputs, **kwargs):
        x = self.conv2a(inputs)
        x = self.bn2a(x)
        x = self.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x)
        x = self.relu(x)

        if self.downsample:
            shortcut = self.conv_shortcut(inputs)
            shortcut = self.bn_shortcut(shortcut)
        else:
            shortcut = inputs

        x = layers.add([x, shortcut])
        x = self.relu(x)

        return x


# for 50, 101 or 152 layers
class Block(Model):

    def __init__(self, filters, block_name, downsample=False, stride=1, **kwargs):
        self.expasion = 4
        super(Block, self).__init__(**kwargs)

        conv_name = 'res' + block_name + '_branch'
        bn_name = 'bn' + block_name + '_branch'
        self.downsample = downsample

        self.conv2a = Conv2D(filters=filters,
                                          kernel_size=1,
                                          strides=stride,
                                          kernel_initializer='he_normal',
                                          name=conv_name + '2a')
        self.bn2a = BatchNormalization(axis=3, name=bn_name + '2a')

        self.conv2b = Conv2D(filters=filters,
                                          kernel_size=3,
                                          padding='same',
                                          kernel_initializer='he_normal',
                                          name=conv_name + '2b')
        self.bn2b = BatchNormalization(axis=3, name=bn_name + '2b')

        self.conv2c = Conv2D(filters=4 * filters,
                                          kernel_size=1,
                                          kernel_initializer='he_normal',
                                          name=conv_name + '2c')
        self.bn2c = BatchNormalization(axis=3, name=bn_name + '2c')

        if self.downsample:
            self.conv_shortcut = Conv2D(filters=4 * filters,
                                                     kernel_size=1,
                                                     strides=stride,
                                                     kernel_initializer='he_normal',
                                                     name=conv_name + '1')
            self.bn_shortcut = BatchNormalization(axis=3, name=bn_name + '1')

    def call(self, inputs, **kwargs):
        x = self.conv2a(inputs)
        x = self.bn2a(x)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x)

        if self.downsample:
            shortcut = self.conv_shortcut(inputs)
            shortcut = self.bn_shortcut(shortcut)
        else:
            shortcut = inputs

        x = layers.add([x, shortcut])
        x = tf.nn.relu(x)

        return x


class ResNet(Model):
    def __init__(self, block, layers, num_classes, method, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.method = method
        #self.padding = ZeroPadding2D((3, 3))
        # self.conv1 = Conv2D(filters=64, kernel_size=3, strides=1,
        #                                  kernel_initializer='glorot_uniform',
        #                                  name='conv1')
        self.conv1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.bn_conv1 = BatchNormalization(axis=3,fused=False, name='bn_conv1')
        self.max_pool = MaxPooling2D((2, 2), strides=2, padding='same')

        # layer2
        self.res2 = self.mid_layer(block, 64, layers[0], stride=1, layer_number=2)

        # layer3
        self.res3 = self.mid_layer(block, 128, layers[1], stride=2, layer_number=3)

        # layer4
        self.res4 = self.mid_layer(block, 256, layers[2], stride=2, layer_number=4)

        # layer5
        self.res5 = self.mid_layer(block, 512, layers[3], stride=2, layer_number=5)
        self.avgpool = GlobalAveragePooling2D(name='avg_pool')
        self.flatten = Flatten()
        # self.fc      = Dense(num_classes, name='result')

        if self.method == 'mdmtr':
            self.fc         = Dense(num_classes, use_bias=False, name='result')
        else:
            self.fc         = Dense(num_classes, name='result')

    def mid_layer(self, block, filter, block_layers, stride=1, layer_number=1):
        layer = Sequential()
        if stride != 1 or block == Block:
            layer.add(block(filters=filter,
                            downsample=True, stride=stride,
                            block_name='{}a'.format(layer_number)))
        else:
            layer.add(block(filters=filter,
                            downsample=False, stride=stride,
                            block_name='{}a'.format(layer_number)))

        for i in range(1, block_layers):
            p = chr(i + ord('a'))
            layer.add(block(filters=filter, block_name='{}'.format(layer_number) + p))

        return layer

    def call(self, inputs, mask, with_softmax=True):
        #x = self.padding(inputs)
        #x = tf.cast(inputs, tf.float32)
        x = self.conv1(inputs)
        x = self.bn_conv1(x)
        x = tf.nn.relu(x)
        x = self.max_pool(x)
        # layer2
        x = self.res2(x)
        # layer3
        x = self.res3(x)
        # layer4
        x = self.res4(x)
        # layer5
        x = self.res5(x)

        x = self.avgpool(x)
        #x = self.flatten(x)
        if self.method == 'mdmtr':
            x = tf.nn.l2_normalize(x, axis=-1)
            x = self.fc(x)
        else:
            x = self.fc(x)
            x = self.masked_softmax(x, mask)
        return x

    def masked_softmax(self, scores, mask):
        scores = scores - tf.tile(tf.reduce_max(scores, axis=(1,), keepdims=True), [1, tf.shape(scores)[1]])
        exp_scores = tf.exp(scores)
        exp_scores *= mask
        exp_sum_scores = tf.reduce_sum(exp_scores, axis=1, keepdims=True)
        return exp_scores / (tf.tile(exp_sum_scores, [1, tf.shape(exp_scores)[1]]) + 1e-7)


def resnet18(class_num, method):
    return ResNet(Basic_Block, [2, 2, 2, 2], num_classes=class_num, method=method)


def resnet38(class_num, method):
    return ResNet(Basic_Block, [3, 4, 6, 3], num_classes=class_num, method=method)


def resnet50(class_num, method):
    return ResNet(Block, [3, 4, 6, 3], num_classes=class_num, method=method)


def resnet101(class_num, method):
    return ResNet(Block, [3, 4, 23, 3], num_classes=class_num, method=method)


def resnet152(class_num, method):
    return ResNet(Block, [3, 8, 36, 3], num_classes=class_num, method=method)


def ConstructNetwork(network, class_num, method):   
    if network == 'mlp':
        return MLPModel(class_num, method=method) 
    if network == 'base':
        return BaseModel(class_num, method=method)
    elif network == 'res18':
        return resnet18(class_num, method=method)