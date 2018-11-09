"""The discriminator of SNGAN."""
import tensorflow as tf
from common.model import ops
from gan.sngan.generator import get_kernel
from model.ops import sn_block, attention
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn, LSTMCell


# def discriminator_resnet(x, labels, df_dim, number_classes, kernel=(3, 3), strides=[(2, 2)], dilations=(1, 1),
#                          pooling='avg', update_collection=None, act=tf.nn.relu, scope_name='Discriminator', reuse=False):
#     with tf.variable_scope(scope_name) as scope:
#         if reuse:
#             scope.reuse_variables()
#         tf.summary.histogram("Input", x, family=scope_name)
#         if x.get_shape().as_list()[1] <= 16:
#             h = act(ops.snconv2d(x, df_dim, (x.get_shape().as_list()[1], 1), update_collection=update_collection,
#                                  name='d_sn_conv_init',
#                                  padding="VALID_W"), name='d_sn_conv_init_act')
#         else:
#             h = x
#             tf.summary.histogram("after_conv_h_1", h, family=scope_name)
#         hidden_dim = df_dim
#         for layer in range(len(strides)):
#             print(h.shape)
#             if layer == 1:
#                 h = attention(h, hidden_dim, sn=True, reuse=reuse)
#                 tf.summary.histogram("attention", h, family=scope_name)
#             block_name = 'd_block{}'.format(layer)
#             hidden_dim = hidden_dim * strides[layer][0]
#             dilation_rate = dilations[0] ** (layer + 1), dilations[1] ** (layer + 1)
#             h = sn_block(h, hidden_dim, block_name, get_kernel(h, kernel), strides[layer],
#                          dilation_rate, update_collection, act, pooling, padding='VALID')
#             tf.summary.histogram(block_name, h, family=scope_name)
#
#         end_block = act(h, name="after_resnet_block")
#         tf.summary.histogram("after_resnet_block", end_block, family=scope_name)
#
#         # h_std = ops.minibatch_stddev_layer(end_block)
#         # tf.summary.histogram(h_std.name, h_std, family=scope_name)
#         # h_std_conv_std = act(ops.snconv2d(h_std, hidden_dim, (1, 3), update_collection=update_collection,
#         #                                  name='minibatch_stddev_stride', padding=None, strides=(1, 3)),
#         #                     name="minibatch_stddev_stride_act")
#         # tf.summary.histogram("after_mini_batch_std", h_std_conv_std, family=scope_name)
#         # h_final_flattened = tf.layers.flatten(h_std_conv_std)
#         h_final_flattened = tf.reduce_sum(end_block, [1, 2])
#         tf.summary.histogram("h_final_flattened", h_final_flattened, family=scope_name)
#         output = ops.snlinear(h_final_flattened, 1, update_collection=update_collection, name='d_sn_linear')
#         tf.summary.histogram("final_output", output, family=scope_name)
#         # output = tf.Print(output, [tf.py_func(
#         #     lambda val, score: print_protein_values(val,score),
#         #     [tf.squeeze(x)[0], output[0]], tf.string)], "seq:")
#         return output, h_final_flattened


def discriminator_resnet(x, labels, df_dim, number_classes, kernel=(3, 3), strides=[(2, 2)], dilations=(1, 1),
                         pooling='avg', update_collection=None, act=tf.nn.relu, scope_name='Discriminator', reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        tf.summary.histogram("Input", x, family=scope_name)
        h = x
        hidden_dim = df_dim
        for layer in range(len(strides)):
            print(h.shape)
            if layer == 1:
                h = attention(h, hidden_dim, sn=True, reuse=reuse)
                tf.summary.histogram("attention", h, family=scope_name)
            block_name = 'd_block{}'.format(layer)
            hidden_dim = hidden_dim * strides[layer][0]
            dilation_rate = dilations[0] ** (layer + 1), dilations[1] ** (layer + 1)
            h = sn_block(h, hidden_dim, block_name, get_kernel(h, kernel), strides[layer],
                         dilation_rate, update_collection, act, pooling, padding='VALID')
            tf.summary.histogram(block_name, h, family=scope_name)

        end_block = act(h, name="after_resnet_block")
        tf.summary.histogram("after_resnet_block", end_block, family=scope_name)

        # h_std = ops.minibatch_stddev_layer(end_block)
        # tf.summary.histogram(h_std.name, h_std, family=scope_name)
        # h_std_conv_std = act(ops.snconv2d(h_std, hidden_dim, (1, 3), update_collection=update_collection,
        #                                  name='minibatch_stddev_stride', padding=None, strides=(1, 3)),
        #                     name="minibatch_stddev_stride_act")
        # tf.summary.histogram("after_mini_batch_std", h_std_conv_std, family=scope_name)
        # h_final_flattened = tf.layers.flatten(h_std_conv_std)
        h_final_flattened = tf.reduce_sum(end_block, [1, 2])
        tf.summary.histogram("h_final_flattened", h_final_flattened, family=scope_name)
        output = ops.snlinear(h_final_flattened, 1, update_collection=update_collection, name='d_sn_linear')
        tf.summary.histogram("final_output", output, family=scope_name)
        # output = tf.Print(output, [tf.py_func(
        #     lambda val, score: print_protein_values(val,score),
        #     [tf.squeeze(x)[0], output[0]], tf.string)], "seq:")
        return output, h_final_flattened

def discriminator_1d(x, labels, df_dim, number_classes, kernel=(3, 3), strides=[(2, 2)], dilations=(1, 1),
                     pooling='avg', update_collection=None, act=tf.nn.relu, scope_name='Discriminator', reuse=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        tf.summary.histogram("Input", x, family=scope_name)
        h = act(ops.snconv2d(x, df_dim, (x.get_shape().as_list()[1], 1), update_collection=update_collection,
                             name='d_sn_conv_init',
                             padding="VALID"), name='d_sn_conv_init_act')
        kernel = (1, 5)
        tf.summary.histogram("after_conv_h_1", h, family=scope_name)
        hidden_dim = df_dim
        for layer in range(len(strides)):
            if layer == 1:
                h = attention(h, hidden_dim, sn=True, reuse=reuse)
                tf.summary.histogram("attention", h, family=scope_name)
            block_name = 'd_block{}'.format(layer)
            hidden_dim = hidden_dim * strides[layer][0]
            # dilation_rate = dilations[0] ** (layer + 1), dilations[1] ** (layer + 1)
            dilation_rate = (1, 1)
            h = sn_block(h, hidden_dim, block_name, kernel, strides[layer],
                         dilation_rate, update_collection, act, pooling, padding='VALID')
            tf.summary.histogram(block_name, h, family=scope_name)

        end_block = act(h, name="after_resnet_block")
        tf.summary.histogram("after_resnet_block", end_block, family=scope_name)

        h_std = ops.minibatch_stddev_layer_v2(end_block)
        tf.summary.histogram(h_std.name, h_std, family=scope_name)
        h_std_conv_std = act(ops.snconv2d(h_std, hidden_dim, (1, 5), update_collection=update_collection,
                                          name='minibatch_stddev_stride', padding=None),
                             name="minibatch_stddev_stride_act")
        tf.summary.histogram("after_mini_batch_std", h_std_conv_std, family=scope_name)
        h_final_flattened = tf.reduce_sum(h_std_conv_std, [1, 2])
        tf.summary.histogram("h_final_flattened", h_final_flattened, family=scope_name)
        output = ops.snlinear(h_final_flattened, 1, update_collection=update_collection, name='d_sn_linear')
        tf.summary.histogram("final_output", output, family=scope_name)
        # output = tf.Print(output, [tf.py_func(
        #     lambda val, score: print_protein_values(val,score),
        #     [tf.squeeze(x)[0], output[0]], tf.string)], "seq:")
        return output, h_final_flattened


def discriminator_conditional(x, labels, df_dim, number_classes, kernel=(3, 3), strides=[(2, 2)], dilations=(1, 1),
                              pooling='avg', update_collection=None, act=tf.nn.leaky_relu, scope_name='Discriminator',
                              reuse=False):
    labels = labels[0]
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        output, h_final_flattened = discriminator_resnet(x, labels, df_dim, number_classes, kernel, strides, dilations,
                                                         pooling, update_collection, act, scope_name, reuse)
        h_labels = ops.sn_embedding(labels, number_classes, h_final_flattened.get_shape().as_list()[-1],
                                    update_collection=update_collection, name='label_embedding')
        tf.summary.histogram("label_embedding", h_labels, family=scope_name)
        multiplied_with_labels = h_final_flattened * h_labels
        tf.summary.histogram("multiplied_with_labels", multiplied_with_labels, family=scope_name)
        reduced_multiplied_with_labels = tf.reduce_sum(multiplied_with_labels, axis=1, keepdims=True)
        tf.summary.histogram("reduced_multiplied_with_labels", reduced_multiplied_with_labels, family=scope_name)
        output += reduced_multiplied_with_labels
        tf.summary.histogram("final_output_with_labels", output, family=scope_name)
        return output, h_final_flattened


def discriminator_rnn(x, labels, df_dim, number_classes, kernel=(3, 3), strides=(2, 2), dilations=(1, 1),
                      pooling='avg', update_collection=None, act=tf.nn.relu, scope_name='Discriminator', reuse=False):
    num_layers = 3
    num_nodes = [int(8 / 2), df_dim, df_dim]
    x = tf.transpose(tf.squeeze(x), perm=[0, 2, 1])

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        # Define LSTM cells
        enc_fw_cells = [LSTMCell(num_nodes[layer], name="fw_" + str(layer)) for layer in range(num_layers)]
        enc_bw_cells = [LSTMCell(num_nodes[layer], name="bw_" + str(layer)) for layer in range(num_layers)]

        # Connect LSTM cells bidirectionally and stack
        (all_states, fw_state, bw_state) = stack_bidirectional_dynamic_rnn(
            cells_fw=enc_fw_cells, cells_bw=enc_bw_cells, inputs=x, dtype=tf.float32)

        # Concatenate results
        for k in range(num_layers):
            if k == 0:
                con_c = tf.concat((fw_state[k].c, bw_state[k].c), 1)
                con_h = tf.concat((fw_state[k].h, bw_state[k].h), 1)
            else:
                con_c = tf.concat((con_c, fw_state[k].c, bw_state[k].c), 1)
                con_h = tf.concat((con_h, fw_state[k].h, bw_state[k].h), 1)

        output = all_states[:, x.get_shape.as_list()[2]]
        output = ops.snlinear(output, 1, update_collection=update_collection, name='d_sn_linear')
    return output, tf.concat((fw_state[2].c, bw_state[2].c), 1)
