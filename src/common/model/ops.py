from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf


def get_padding(kernel, dilations, axis):
    """
    Calculates required padding for given axis
    Args:
        kernel: A tuple of kernel height and width
        dilations: A tuple of dilation height and width
        axis: 0 - height, 1 width

    Returns:
        An array that contains a length of padding at the begging and at the end
    """
    extra_padding = (kernel[axis] - 1) * (dilations[axis])
    return [extra_padding // 2, extra_padding - (extra_padding // 2)]


def apply_padding(x, kernel, dilations, padding):
    """
    Adds padding to the edges of tensor such that after applying VALID conv the size of tensor remains the same.
    Acts similar to the SAME conv but instead of 0 padding it using REFLECT
    Args:
      x: Input tensor
      kernel: A tuple of kernel height and width
      dilations: A tuple of dilation height and width
      padding: Type of padding

    Returns:
        Padded tensor
    """
    height_padding = [0, 0]
    width_padding = [0, 0]
    apply = False
    if padding == 'VALID_H' or padding == 'VALID':
        height_padding = get_padding(kernel, dilations, 0)
        apply = True
    if padding == 'VALID_W' or padding == 'VALID':
        width_padding = get_padding(kernel, dilations, 1)
        apply = True
    if padding is None:
        padding = "VALID"
    if apply:
        x = tf.pad(x, [[0, 0], height_padding, width_padding, [0, 0]], "REFLECT")
        padding = 'VALID'
    return x, padding


def conv2d(input_, output_dim, kernel=(3, 3), strides=(1, 1), dilations=(1, 1), name='conv2d', update_collection=None,
           padding='SAME'):
    """Creates convolutional layers which use xavier initializer.

    Args:
      input_: 4D input tensor (batch size, height, width, channel).
      output_dim: Number of features in the output layer.
      kernel: The strides width of the convolutional kernel. (Default value = (3)
      strides: The strides width of the convolutional strides. (Default value = (1)
      dilations: The width and height stride of the convolutional dilations. (Default value = (1)
      name: The name of the variable scope. (Default value = 'conv2d')
      update_collection: The update collections used in the in the spectral_normed_weight. (Default value = None)
      padding:  (Default value = 'SAME')

    Returns:
      conv: 2D conv layer.

    """
    with tf.variable_scope(name):
        input_, padding = apply_padding(input_, kernel, strides, dilations, padding)
        w = tf.get_variable('w', [kernel[0], kernel[1], input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = _conv2d(input_, w, output_dim, strides, dilations, padding)
        return conv


def deconv2d(input_, output_shape, kernel=(3, 3), strides=(1, 1), dilations=(1, 1), stddev=0.02,
             name='deconv2d', init_bias=0.):
    """Creates deconvolutional layers.

    Args:
      input_: 4D input tensor (batch size, height, width, channel).
      output_shape: Number of features in the output layer.
      kernel: The strides width of the convolutional kernel. (Default value = (3)
      strides: The strides width of the convolutional strides. (Default value = (1)
      dilations: The width and height stride of the convolutional dilations. (Default value = (1)
      stddev: The standard deviation for weights initializer. (Default value = 0.02)
      name: The name of the variable scope. (Default value = 'deconv2d')
      init_bias:   (Default value = 0.)

    Returns:
      conv: 2D deconv layer.

    """
    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [kernel[0], kernel[1], output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, strides[0], strides[1], 1])
        biases = tf.get_variable('biases', [output_shape[-1]],
                                 initializer=tf.constant_initializer(init_bias))
        deconv = tf.nn.bias_add(deconv, biases)
        deconv.shape.assert_is_compatible_with(output_shape)

        return deconv


def linear(x, output_size, scope=None, bias_start=0.0):
    """Creates a linear layer.

    Args:
      x: 2D input tensor (batch size, features)
      output_size: Number of features in the output layer
      scope: Optional, variable scope to put the layer's parameters into (Default value = None)
      bias_start: The bias parameters are initialized to this value (Default value = 0.0)

    Returns:
      Linear transformation of input x

    """
    shape = x.get_shape().as_list()

    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable('Matrix', [shape[1], output_size], tf.float32, tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(bias_start))
    out = tf.matmul(x, matrix) + bias
    return out


def l2normalize(v, eps=1e-12):
    """l2 normalize the input vector.

    Args:
      v: tensor to be normalized
      eps:  epsilon (Default value = 1e-12)

    Returns:
      A normalized tensor
    """
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(weights, num_iters=1, update_collection=None, with_sigma=False):
    """Performs Spectral Normalization on a weight tensor.
    
    Specifically it divides the weight tensor by its largest singular value. This
    is intended to stabilize GAN training, by making the discriminator satisfy a
    local 1-Lipschitz constraint.
    Based on [Spectral Normalization for Generative Adversarial Networks][sn-model]
    [sn-model] https://openreview.net/pdf?id=B1QRgziT-

    Args:
      weights: The weight tensor which requires spectral normalization
      num_iters: Number of SN iterations. (Default value = 1)
      update_collection: The update collection for assigning persisted variable u.
    If None, the function will update u during the forward
    pass. Else if the update_collection equals 'NO_OPS', the
    function will not update the u during the forward. This
    is useful for the discriminator, since it does not update
    u in the second pass.
    Else, it will put the assignment in a collection
    defined by the user. Then the user need to run the
    assignment explicitly. (Default value = None)
      with_sigma: For debugging purpose. If True, the fuction returns (Default value = False)

    Returns:
      w_bar: The normalized weight tensor
      sigma: The estimated singular value for the weight tensor.

    """
    w_shape = weights.shape.as_list()
    w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
    u = tf.get_variable('u', [1, w_shape[-1]],
                        initializer=tf.truncated_normal_initializer(),
                        trainable=False)
    u_ = u
    for _ in range(num_iters):
        v_ = l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = l2normalize(tf.matmul(v_, w_mat))

    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat /= sigma
    if update_collection is None:
        with tf.control_dependencies([u.assign(u_)]):
            w_bar = tf.reshape(w_mat, w_shape)
    else:
        w_bar = tf.reshape(w_mat, w_shape)
        if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(u_))
    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar


def snconv2d(input_, output_dim, kernel=(3, 3), strides=(1, 1), dilations=(1, 1), sn_iters=1, update_collection=None,
             name='snconv2d', padding='SAME'):
    """Creates a spectral normalized (SN) convolutional layer.

    Args:
      input_: 4D input tensor (batch size, height, width, channel).
      output_dim: Number of features in the output layer.
      kernel: The height and width of the convolutional kernel. (Default value = (3)
      strides: The height and width of the convolutional strides. (Default value = (1)
      dilations: The height and width of the convolutional dilations. (Default value = (1)
      sn_iters: The number of SN iterations. (Default value = 1)
      update_collection: The update collection used in spectral_normed_weight. (Default value = None)
      name: Variable scope name (Default value = 'snconv2d')
      padding:  (Default value = 'SAME')

    Returns:
      A conv layer with spectral normalization

    """
    with tf.variable_scope(name):
        input_, padding = apply_padding(input_, kernel, dilations, padding)
        w = tf.get_variable(
            'w', [kernel[0], kernel[1], input_.get_shape()[-1], output_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)
        conv = _conv2d(input_, w_bar, output_dim, strides, dilations, padding)
        return conv


def sndeconv2d(input_, output_shape, kernel=(3, 3), strides=(1, 1), dilations=(1, 1), sn_iters=1,
               update_collection=None, name='deconv2d'):
    """Creates a spectral normalized (SN) deconvolutional layer.

    Args:
      input_: 4D input tensor (batch size, height, width, channel).
      output_shape: Number of features in the output layer.
      kernel: The strides width of the convolutional kernel. (Default value = (3)
      strides: The strides width of the convolutional strides. (Default value = (1)
      dilations: The width and height stride of the convolutional dilations. (Default value = (1)
      stddev: The standard deviation for weights initializer. (Default value = 0.02)
      sn_iters:  The number of SN iterations. (Default value = 1)
      update_collection: The update collection used in spectral_normed_weight. (Default value = None)
      name: The name of the variable scope. (Default value = 'deconv2d')

    Returns:
      conv: The normalized tensor.

    """
    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [kernel[0], kernel[1], output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.contrib.layers.xavier_initializer())
        w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)
        deconv = tf.nn.conv2d_transpose(input_, w_bar, output_shape=output_shape,
                                        strides=[1, strides[0], strides[1], 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.zeros_initializer())
        deconv = tf.nn.bias_add(deconv, biases)
        deconv.shape.assert_is_compatible_with(output_shape)

        return deconv


def snlinear(x, output_size, bias_start=0.0, sn_iters=1, update_collection=None, name='snlinear'):
    """Creates a spectral normalized linear layer.

    Args:
      x: 2D input tensor (batch size, features).
      output_size: Number of features in output of layer.
      bias_start: The bias parameters are initialized to this value (Default value = 0.0)
      sn_iters: Number of SN iterations. (Default value = 1)
      update_collection: The update collection used in spectral_normed_weight (Default value = None)
      name:  (Default value = 'snlinear')

    Returns:
      The linear transoformation of x with spectral normalization

    """
    shape = x.get_shape().as_list()

    with tf.variable_scope(name):
        matrix = tf.get_variable('Matrix', [shape[1], output_size], tf.float32, tf.contrib.layers.xavier_initializer())
        matrix_bar = spectral_normed_weight(matrix, num_iters=sn_iters, update_collection=update_collection)
        bias = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(bias_start))
        out = tf.matmul(x, matrix_bar) + bias
        return out


def sn_embedding(x, number_classes, embedding_size, sn_iters=1,
                 update_collection=None, name='snembedding', embedding_map_name = 'embedding_map'):
    """Creates a spectral normalized embedding lookup layer.

    Args:
      x: 1D input tensor (batch size, ).
      number_classes: The number of classes.
      embedding_size: The length of the embeddding vector for each class.
      sn_iters: Number of SN iterations. (Default value = 1)
      update_collection: The update collection used in spectral_normed_weight (Default value = None)
      name: Scope name (Default value = 'snembedding')
      embedding_map_name: The name of embedding (Default value = 'embedding_map')

    Returns:
      The output tensor (batch size, embedding_size).

    """
    with tf.variable_scope(name):
        embedding_map = tf.get_variable(name=embedding_map_name, shape=[number_classes, embedding_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
        embedding_map_bar_transpose = spectral_normed_weight(tf.transpose(embedding_map), num_iters=sn_iters,
                                                             update_collection=update_collection)
        embedding_map_bar = tf.transpose(embedding_map_bar_transpose)
        return tf.nn.embedding_lookup(embedding_map_bar, x)


class ConditionalBatchNorm(object):
    """Conditional BatchNorm.
    
    For each  class, it has a specific gamma and beta as normalization variable.

    Args:
        num_categories: A number of different classes
        name: A name of the scope (Default value = 'conditional_batch_norm')
        center: (Default value = True)
        scale: (Default value = True)

    Returns:
        Conditional Batch Norm layer
    """

    def __init__(self, num_categories, name='conditional_batch_norm', center=True, scale=True):
        with tf.variable_scope(name):
            self.name = name
            self.num_categories = num_categories
            self.center = center
            self.scale = scale

    def __call__(self, inputs, labels):
        inputs = tf.convert_to_tensor(inputs)
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        axis = [0, 1, 2]
        shape = tf.TensorShape([self.num_categories]).concatenate(params_shape)

        with tf.variable_scope(self.name):
            self.gamma = tf.get_variable('gamma', shape, initializer=tf.ones_initializer())
            self.beta = tf.get_variable('beta', shape, initializer=tf.zeros_initializer())
            beta = tf.gather(self.beta, labels)
            beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)
            gamma = tf.gather(self.gamma, labels)
            gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
            mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)
            variance_epsilon = 1E-5
            outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, variance_epsilon)
            outputs.set_shape(inputs_shape)
            return outputs


class BatchNorm(object):
    """The Batch Normalization layer.

    Args:
        name: a name of the scope (Default value = 'batch_norm')
        center: (Default value = True)
        scale: (Default value = True)

    Returns:
        Batch Norm layer
    """

    def __init__(self, name='batch_norm', center=True, scale=True):
        with tf.variable_scope(name):
            self.name = name
            self.center = center
            self.scale = scale

    def __call__(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        inputs_shape = inputs.get_shape().as_list()
        params_shape = inputs_shape[-1]
        axis = [0, 1, 2]
        shape = tf.TensorShape([params_shape])

        with tf.variable_scope(self.name):
            self.gamma = tf.get_variable('gamma', shape, initializer=tf.ones_initializer())
            self.beta = tf.get_variable('beta', shape, initializer=tf.zeros_initializer())
            beta = self.beta
            gamma = self.gamma
            mean, variance = tf.nn.moments(inputs, axis, keep_dims=True)
            variance_epsilon = 1E-5
            outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, variance_epsilon)
            outputs.set_shape(inputs_shape)
            return outputs


def _block(x, out_channels, name, conv=conv2d, kernel=(3, 3), strides=(2, 2), dilations=(1, 1), update_collection=None,
           act=tf.nn.leaky_relu, pooling='avg', padding='SAME', batch_norm=False):
    """Builds the residual blocks used in the discriminator in GAN.

    Args:
      x: The 4D input vector.
      out_channels: Number of features in the output layer.
      name: The variable scope name for the block.
      conv: Convolution function. Options conv2d or snconv2d
      kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
      strides: Rate of convolution strides (Default value = (2, 2))
      dilations: Rate of convolution dilation (Default value = (1, 1))
      update_collection: The update collections used in the in the spectral_normed_weight. (Default value = None)
      downsample: If True, downsample the spatial size the input tensor .
      If False, the spatial size of the input tensor is unchanged. (Default value = True)
      act: The activation function used in the block. (Default value = tf.nn.relu)
      pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
      padding: Padding type (Default value = 'SAME')
      batch_norm: A flag that determines if batch norm should be used (Default value = False)

    Returns:
      A tensor representing the output of the operation.

    """
    with tf.variable_scope(name):
        if batch_norm:
            bn0 = BatchNorm(name='bn_0')
            bn1 = BatchNorm(name='bn_1')
        input_channels = x.shape.as_list()[-1]
        x_0 = x
        x = conv(x, out_channels, kernel, dilations=dilations, name='conv1', padding=padding)
        if batch_norm:
            x = bn0(x)
        x = act(x, name="before_downsampling")
        x = down_sampling(x, conv, pooling, out_channels, kernel, strides, update_collection, 'conv2', padding)
        if batch_norm:
            x = bn1(x)
        if strides[0] > 1 or strides[1] > 1 or input_channels != out_channels:
            x_0 = down_sampling(x_0, conv, pooling, out_channels, kernel, strides, update_collection, 'conv3',
                                padding)
        out = x_0 + x  # No RELU: http://torch.ch/blog/2016/02/04/resnets.html
        return out


def block(x, out_channels, name, kernel=(3, 3), strides=(1, 1), dilations=(1, 1), update_collection=None,
          act=tf.nn.leaky_relu, pooling='avg', padding='SAME'):
    """Builds the residual blocks used in GAN. It used standard 2D conv.

    Args:
      x: The 4D input vector.
      out_channels: Number of features in the output layer.
      name: The variable scope name for the block.
      kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
      strides: The height and width of convolution strides (Default value = (1, 1))
      dilations: The height and width of convolution dilation (Default value = (1, 1))
      update_collection: The update collections used in the spectral_normed_weight. (Default value = None)
      act: The activation function used in the block. (Default value = tf.nn.leaky_relu)
      pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
      If False, the spatial size of the input tensor is unchanged. (Default value = True)
      padding:  Type of padding (Default value = 'SAME')

    Returns:
      A Tensor representing the output of the operation.

    """
    return _block(x, out_channels, name, conv2d, kernel, strides, dilations, update_collection,
                  act, pooling, padding, False)


def block_norm(x, out_channels, name, kernel=(3, 3), strides=(1, 1), dilations=(1, 1), update_collection=None,
               act=tf.nn.leaky_relu, pooling='avg', padding='SAME'):
    """Builds the residual blocks used in GAN. It used standard 2D conv and batch norm.

    Args:
      x: The 4D input vector.
      out_channels: Number of features in the output layer.
      name: The variable scope name for the block.
      kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
      strides: The height and width of convolution strides (Default value = (1, 1))
      dilations: The height and width of convolution dilation (Default value = (1, 1))
      update_collection: The update collections used in the spectral_normed_weight. (Default value = None)
      act: The activation function used in the block. (Default value = tf.nn.leaky_relu)
      pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
      If False, the spatial size of the input tensor is unchanged. (Default value = True)
      padding:  Type of padding (Default value = 'SAME')

    Returns:
      A Tensor representing the output of the operation.

    """
    return _block(x, out_channels, name, conv2d, kernel, strides, dilations, update_collection,
                  act, pooling, padding, True)


def sn_block(x, out_channels, name, kernel=(3, 3), strides=(1, 1), dilations=(1, 1), update_collection=None,
             act=tf.nn.leaky_relu, pooling='avg', padding='SAME'):
    """Builds the residual blocks used in SNGAN. It used 2D conv with spectral normalization.

    Args:
      x: The 4D input vector.
      out_channels: Number of features in the output layer.
      name: The variable scope name for the block.
      kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
      strides: The height and width of convolution strides (Default value = (1, 1))
      dilations: The height and width of convolution dilation (Default value = (1, 1))
      update_collection: The update collections used in the spectral_normed_weight. (Default value = None)
      act: The activation function used in the block. (Default value = tf.nn.leaky_relu)
      pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
      If False, the spatial size of the input tensor is unchanged. (Default value = True)
      padding:  Type of padding (Default value = 'SAME')

    Returns:
      A Tensor representing the output of the operation.

    """
    return _block(x, out_channels, "sn_" + name, snconv2d, kernel, strides, dilations, update_collection,
                  act, pooling, padding, False)


def sn_norm_block(x, out_channels, name, kernel=(3, 3), strides=(1, 1), dilations=(1, 1), update_collection=None,
                  act=tf.nn.leaky_relu, pooling='avg', padding='SAME'):
    """Builds the residual blocks used in SNGAN. . It used 2D conv with spectral normalization and batchnorm

    Args:
      x: The 4D input vector.
      out_channels: Number of features in the output layer.
      name: The variable scope name for the block.
      kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
      strides: The height and width of convolution strides (Default value = (1, 1))
      dilations: The height and width of convolution dilation (Default value = (1, 1))
      update_collection: The update collections used in the spectral_normed_weight. (Default value = None)
      act: The activation function used in the block. (Default value = tf.nn.leaky_relu)
      pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
      downsample: If True, downsample the spatial size the input tensor.
      If False, the spatial size of the input tensor is unchanged. (Default value = True)
      padding:  Type of padding (Default value = 'SAME')

    Returns:
      A Tensor representing the output of the operation.

    """
    return _block(x, out_channels, "sn_b_" + name, snconv2d, kernel, strides, dilations, update_collection,
                  act, pooling, padding, True)


def down_sampling(x, covn_fn, pooling, out_channels, kernel, strides, update_collection, name, padding='SAME'):
    """
        Performs convolution plus downsamping if required
    Args:
      x: tensor input
      covn_fn: function that performs convulutions
      If False, the spatial size of the input tensor is unchanged.
      pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
      out_channels: Number of features in the output layer.
      kernel: The height and width of the convolution kernel filter
      strides: Rate of convolution strides
      update_collection: The update collections used in the in the spectral_normed_weight. (Default value = None)
      name:  Variable scope name
      padding:  Padding type (Default value = 'SAME')

    Returns:
        A tensor representing the output of the operation.
    """
    if pooling == 'avg':
        x = covn_fn(x, out_channels, kernel, update_collection=update_collection, name=name, padding=padding)
        x = tf.nn.avg_pool(x, [1, strides[0], strides[1], 1], [1, strides[0], strides[1], 1], 'VALID')
    else:
        x = covn_fn(x, out_channels, kernel, strides, update_collection=update_collection, name=name, padding=padding)
    return x


def up_sample(x, multipliers):
    """
        Upsamples given input
    Args:
      x: Input tensor
      multipliers: A tuple of two ints. First one - factor by which to increase height.
      Second one - factor by which to increase width.

    Returns:
        An upsampled tensor
    """
    _, nh, nw, nx = x.get_shape().as_list()
    x = tf.image.resize_nearest_neighbor(x, [nh * multipliers[0], nw * multipliers[1]])
    return x


def _conv2d(x, kernel, output_dim, strides, dilations, padding):
    """Performs convolution and adds bias

    Args:
      x: input tensor
      kernel: Convolution kernel
      output_dim: Output dimentions (number of filters)
      strides: Height and width of strides used in convolution
      dilations: Height and width of dilations used in convolution
      padding: Type of padding: 'SAME' or 'VALID'

    Returns:
      Results after applying convolutions and adding bias
    """
    conv = tf.nn.conv2d(x, kernel, strides=[1, strides[0], strides[1], 1],
                        dilations=[1, dilations[0], dilations[1], 1], padding=padding)
    biases = tf.get_variable('biases', [output_dim], initializer=tf.zeros_initializer())
    conv = tf.nn.bias_add(conv, biases)
    return conv


def _phase_shift(x, r, axis=1):
    """ Helper function with main phase shift operation

    Args:
      x: Tensor to upsample:
      r: Rate to upsample chosen axis
      axis: Axis to upsample (Default value = 1)

    Returns:
        Upsampled input
    """

    bsize, a, b, c = x.get_shape().as_list()
    assert c % r == 0
    remainder = int(c / r)
    X = tf.reshape(x, (-1, a, b, r, remainder))
    X = tf.split(X, r, axis=3)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x) for x in X], axis=axis)  # bsize, b, a*r, r
    if axis == 1:
        a = a * r
    elif axis == 2:
        b = b * r
    else:
        raise NotImplementedError
    return tf.reshape(X, (-1, a, b, remainder))


def upsample_ps(x, r=(1, 1)):
    """
    Performs phase shift upsampling for width and height
    Args:
      x: Tensor to upsample
      r: A tuple of upsample factors(height and width) (Default value = (1)
    Returns:
      Tensor with height of original_height*r[0] and width original_width*r[1]

    """
    x = _phase_shift(x, r[1], axis=2)
    x = _phase_shift(x, r[0], axis=1)
    return x


def cycle_padding(x, axis=1):
    """

    Args:
      x: Input tensor
      axis: The axis along which cycle padding needs to be applied (Default value = 1)

    Returns:
      A tensor that for chosen axis was padded with data from different end - creating a cycle

    """

    middle_point = x.get_shape().as_list()[axis] // 2

    if axis == 1:
        prefix = x[:, middle_point + 1:, :, :]
        suffix = x[:, :middle_point, :, :]
    elif axis == 2:
        prefix = x[:, :, :, middle_point + 1:]
        suffix = x[:, :, :, :middle_point]
    else:
        raise NotImplementedError
    return tf.concat([prefix, x, suffix], axis=axis)


# def minibatch_stddev_layer(x, group_size=4):
#     with tf.variable_scope('MinibatchStddev'):
#         group_size = tf.minimum(group_size,
#                                 tf.shape(x)[0])  # Minibatch must be divisible by (or smaller than) group_size.
#         s = x.shape  # [NCHW]  Input shape.
#         y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])  # [GMCHW] Split minibatch into M groups of size G.
#         y = tf.cast(y, tf.float32)  # [GMCHW] Cast to FP32.
#         y -= tf.reduce_mean(y, axis=0, keepdims=True)  # [GMCHW] Subtract mean over group.
#         y = tf.reduce_mean(tf.square(y), axis=0)  # [MCHW]  Calc variance over group.
#         y = tf.sqrt(y + 1e-8)  # [MCHW]  Calc stddev over group.
#         y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)  # [M111]  Take average over fmaps and pixels.
#         y = tf.cast(y, x.dtype)  # [M111]  Cast back to original data type.
#         y = tf.tile(y, [group_size, s[1], s[2], 1])  # [N1HW]  Replicate over group and pixels.
#         return tf.concat([x, y], axis=3)  # [NHWC]  Append as new fmap.

def minibatch_stddev_layer(x, group_size=4):
    """
        Original version from ProGAN
    Args:
      x: Input tensor
      group_size:  The number of groups (Default value = 4)

    Returns:
        A standard deviation of chosen number groups. This result is repeated until the shape is matching input
        shape for concatication

    TODO:
        It contains bugs, however, it works better than fixed version. Needs some investigation
    """
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size,
                                tf.shape(x)[0])  # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape  # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])  # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)  # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)  # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)  # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)  # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)  # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)  # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])  # [N1HW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)  # [NHW]  Append as new fmap.


def minibatch_stddev_layer_v2(x, group_size=1):
    """
        Simplified version of ProGAN minibatch discriminator for 1D
    Args:
      x: Input tensor
      group_size:  The number of groups (Default value = 4)

    Returns:
        A standard deviation of chosen number groups. This result is repeated until the shape is matching input
        shape for concatication
    """
    with tf.variable_scope('MinibatchStddev'):
        batch_size = tf.shape(x)[0]
        y = x
        y -= tf.reduce_mean(y, axis=[0, 1, 3], keepdims=True)  # [NCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=[0, 1, 3], keepdims=True)  # [11W1]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)  # [11W1]  Calc stddev over group..
        y = tf.tile(y, [batch_size, 1, 1, 1])  # [N1W1]  Replicate over group and pixels.
        return tf.concat([x, y], axis=3)  # [NHWC]  Append as new fmap.


def hw_flatten(x):
    """
        Flattens along the height and width of the tensor
    Args:
      x: Input tensor

    Returns:
        Flattened tensor
    """
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])


def attention(x, ch, sn=True, scope='attention', reuse=False):
    """
        Self-Attention layer
    Args:
      x: Input tensor
      ch: Number of output channels
      sn:  Flag to decide whether to use conv with spectral normalization (Default value = True)
      scope:  Variable scope name (Default value = 'attention')
      reuse:  Flag to identify whether to reuse variables (Default value = False)

    Returns:
        A tensor representing the output of the operation.
    """
    with tf.variable_scope(scope, reuse=reuse):
        assert ch % 2 == 0

        ch_sqrt = int(math.sqrt(ch))
        if sn:
            f = snconv2d(x, ch_sqrt, kernel=(1, 1), name='sn_conv_f')
            g = snconv2d(x, ch_sqrt, kernel=(1, 1), name='sn_conv_g')
            h = snconv2d(x, ch, kernel=(1, 1), name='sn_conv_h')
        else:
            f = conv2d(x, ch_sqrt, kernel=(1, 1), name='conv_f')
            g = conv2d(x, ch_sqrt, kernel=(1, 1), name='conv_g')
            h = conv2d(x, ch, kernel=(1, 1), name='conv_h')

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
        x = gamma * o + x

    return x


def pad_up_to(x, output_shape, constant_values=0, dynamic_padding=False):
    """

    Args:
      x: Input tensor
      output_shape: Output shape
      constant_values:  Values used for padding (Default value = 0)

    Returns:
        Returns padded tensor that is the shape of output_shape.
    """
    s = tf.shape(x)
    paddings = [ calculate(s[i], m, dynamic_padding) for (i, m) in enumerate(output_shape)]
    return tf.pad(x, paddings, 'CONSTANT', constant_values=constant_values)

def calculate(current_shape, target_shape, dynamic_padding=False):
    if dynamic_padding:
        missing_padding = target_shape - current_shape

        def empty_padding(): return [missing_padding, missing_padding]

        def random_padding():
            front = tf.squeeze(
                tf.random_uniform([1], minval=0, maxval=missing_padding, name="random_padding", dtype=tf.int32))
            end = missing_padding - front
            return [front, end]

        return tf.cond(tf.equal(missing_padding, tf.constant(0)),
                       empty_padding,
                       random_padding)

    else:
        return [0, target_shape - current_shape]








