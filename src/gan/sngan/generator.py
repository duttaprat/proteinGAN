"""The generator of SNGAN."""

import tensorflow as tf
from bio.constants import SMILES_CHARACTER_TO_ID
from common.model import ops
from model.generator_ops import get_kernel, get_dimentions_factors, sn_block_conditional, sn_block
from model.ops import attention
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn, LSTMCell


# def generator_resnet(zs, labels, gf_dim, num_classes, kernel=(3, 3), strides=[(2, 2)], dilations=(1, 1),
#                      pooling='avg', scope_name='Generator', reuse=False, output_shape=[8, 128, 1], act=tf.nn.leaky_relu):
#     with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
#
#         height_d, width_d = get_dimentions_factors(strides)
#
#         number_of_layers = len(strides)
#         hidden_dim = gf_dim * (2 ** (number_of_layers - 1))
#
#         c_h = int(output_shape[0] / height_d)
#         c_w = int((output_shape[1] / width_d) / 2)
#         h = ops.snlinear(zs, c_h * int(c_w) * int((2 * hidden_dim) / (c_h * c_w)), name='noise_linear')
#         tf.summary.histogram("noise_snlinear", h, family=scope_name)
#         h = tf.reshape(h, shape=[-1, c_h, int(c_w), int((2 * hidden_dim) / (c_h * c_w))])
#         new_shape = [h.get_shape().as_list()[0], h.get_shape().as_list()[1],
#                      h.get_shape().as_list()[2] * 2, hidden_dim]
#         h = ops.sndeconv2d(h, new_shape, (c_h, 3), name='noise_expand', strides=(1, 2))
#         h = act(h)
#         tf.summary.histogram("before_blocks", h, family=scope_name)
#         print("COMPRESSED TO: ", h.shape)
#
#         with tf.variable_scope("up", reuse=tf.AUTO_REUSE):
#             for layer_id in range(number_of_layers):
#                 block_name = 'up_block{}'.format(number_of_layers - (layer_id + 1))
#                 dilation_rate = (dilations[0] ** (number_of_layers - layer_id),
#                                  dilations[1] ** (number_of_layers - layer_id))
#                 h = sn_block(h, hidden_dim, block_name, get_kernel(h, kernel),
#                              strides[layer_id], dilation_rate, act, pooling, 'VALID')
#                 tf.summary.histogram(block_name, h, family=scope_name)
#                 if layer_id == number_of_layers - 2:
#                     h = attention(h, hidden_dim, sn=True, reuse=reuse)
#                     tf.summary.histogram("up_attention", h, family=scope_name)
#                 hidden_dim = hidden_dim / strides[layer_id][1]
#
#         bn = ops.BatchNorm(name='g_bn')
#         h_act = tf.nn.leaky_relu(bn(h), name="h_act")
#         if output_shape[2] == 1:
#             out = tf.nn.tanh(ops.snconv2d(h_act, 1, (output_shape[0], 1), name='last_conv'), name="generated")
#         else:
#             out = tf.nn.tanh(ops.snconv2d(h_act, 3, (1, 1), name='last_conv'), name="generated")
#         tf.summary.histogram("Generated_results", out, family=scope_name)
#         print("GENERATED SHAPE", out.shape)
#         return out


def generator_resnet(zs, labels, gf_dim, num_classes, kernel=(3, 3), strides=[(2, 2)], dilations=(1, 1),
                     pooling='avg', scope_name='Generator', reuse=False, output_shape=[8, 128, 1], act=tf.nn.leaky_relu):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        height_d, width_d = get_dimentions_factors(strides)
        number_of_layers = len(strides)
        hidden_dim = gf_dim * (2 ** (number_of_layers-1))

        c_h = int(output_shape[0] / height_d)
        c_w = int((output_shape[1] / width_d))
        h = ops.snlinear(zs, c_h * c_w * hidden_dim, name='noise_linear')
        h = tf.reshape(h, [-1, c_h, c_w, hidden_dim])
        print("COMPRESSED TO: ", h.shape)

        with tf.variable_scope("up", reuse=tf.AUTO_REUSE):
            for layer_id in range(number_of_layers):
                print(h.shape)
                block_name = 'up_block{}'.format(number_of_layers - (layer_id + 1))
                dilation_rate = (dilations[0] ** (number_of_layers - layer_id),
                                 dilations[1] ** (number_of_layers - layer_id))
                h = sn_block(h, hidden_dim, block_name, get_kernel(h, kernel),
                             strides[layer_id], dilation_rate, act, pooling, 'VALID')
                tf.summary.histogram(block_name, h, family=scope_name)
                if layer_id == number_of_layers - 2:
                    h = attention(h, hidden_dim, sn=True, reuse=reuse)
                    tf.summary.histogram("up_attention", h, family=scope_name)
                hidden_dim = hidden_dim / strides[layer_id][1]

        bn = ops.BatchNorm(name='g_bn')
        h_act = tf.nn.leaky_relu(bn(h), name="h_act")
        if output_shape[2] == 1:
            out = tf.nn.tanh(ops.snconv2d(h_act, 1, (output_shape[0], 1), name='last_conv'), name="generated")
        else:
            out = tf.nn.tanh(ops.snconv2d(h_act, 3, (1, 1), name='last_conv'), name="generated")
        tf.summary.histogram("Generated_results", out, family=scope_name)
        print("GENERATED SHAPE", out.shape)
        return out

def generator_1d(zs, labels, gf_dim, num_classes, kernel=(3, 3), strides=(2, 2), dilations=(1, 1),
                 pooling='avg', scope_name='Generator', reuse=False, output_shape=[8, 128, 1], act=tf.nn.leaky_relu):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        # strides_schedule = get_strides_schedule(input_width)
        strides_schedule = [(1, 2), (1, 2), (1, 2), (1, 2)]
        height_d, width_d = get_dimentions_factors(strides_schedule)

        number_of_layers = len(strides_schedule)
        hidden_dim = gf_dim * (2 ** (number_of_layers - 1))
        kernel = (1, 5)
        c_h = 1  # int(embedding_height / height_d)
        c_w = int((output_shape[1] / width_d))
        h = ops.snlinear(zs, c_h * int(c_w) * hidden_dim, name='noise_linear')
        tf.summary.histogram("noise_snlinear", h, family=scope_name)
        h = tf.reshape(h, shape=[-1, c_h, int(c_w), hidden_dim])
        # new_shape = [h.get_shape().as_list()[0], h.get_shape().as_list()[1],
        #              h.get_shape().as_list()[2] * 2, hidden_dim]
        # h = ops.sndeconv2d(h, new_shape, (c_h, 3), name='noise_expand', strides=(1, 2))
        # h = act(h)
        tf.summary.histogram("before_blocks", h, family=scope_name)
        print("COMPRESSED TO: ", h.shape)

        with tf.variable_scope("up", reuse=tf.AUTO_REUSE):
            for layer_id in range(number_of_layers):
                block_name = 'up_block{}'.format(number_of_layers - (layer_id + 1))
                # dilation_rate = (dilations[0] ** (number_of_layers - layer_id),
                #                  dilations[1] ** (number_of_layers - layer_id))
                dilation_rate = (1, 1)
                h = sn_block(h, hidden_dim, block_name, kernel,
                             strides_schedule[layer_id], dilation_rate, act, pooling, 'VALID')
                tf.summary.histogram(block_name, h, family=scope_name)
                if layer_id == number_of_layers - 2:
                    h = attention(h, hidden_dim, sn=True, reuse=reuse)
                    tf.summary.histogram("up_attention", h, family=scope_name)
                hidden_dim = hidden_dim / 2

        bn = ops.BatchNorm(name='g_bn')
        h_act = tf.nn.leaky_relu(bn(h), name="h_act")
        h_act = tf.reshape(h_act, [-1, output_shape[0], output_shape[1], int(hidden_dim / output_shape[0]) * 2])
        h_act = ops.snconv2d(h_act, int(hidden_dim / output_shape[0]) * 2, (output_shape[0], 1),
                             name='embedding_height')
        out = tf.nn.tanh(ops.snconv2d(h_act, 1, (output_shape[0], 1), name='last_conv'), name="generated")
        tf.summary.histogram("Generated_results", out, family=scope_name)
        print("GENERATED SHAPE", out.shape)
        return out


# def generator_conditional(zs, labels, gf_dim, num_classes, kernel=(3, 3), strides=[(2, 2)], dilations=(1, 1),
#                           pooling='avg', scope_name='Generator', reuse=False, output_shape=[8, 128, 1],
#                           act=tf.nn.leaky_relu):
#     with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
#
#         reactions = labels[1]
#         labels = labels[0]
#
#         compressed_reaction = reaction_encoder(dilations, gf_dim, kernel, reactions, pooling, reuse, scope_name,
#                                                output_shape[1])
#
#         height_d, width_d = get_dimentions_factors(strides)
#
#         number_of_layers = len(strides)
#
#         c_h = int(output_shape[0] / height_d)
#         c_w = int((output_shape[1] / width_d))
#         h = ops.snlinear(zs, c_h * int(c_w) * compressed_reaction.get_shape().as_list()[3], name='noise_linear')
#         tf.summary.histogram("noise_snlinear", h, family=scope_name)
#         h = tf.reshape(h, shape=[-1, c_h, int(c_w), compressed_reaction.get_shape().as_list()[3]])
#         tf.summary.histogram("before_blocks", h, family=scope_name)
#         # new_shape = [h.get_shape().as_list()[0], h.get_shape().as_list()[1],
#         #              h.get_shape().as_list()[2] * 2, int(hidden_dim / 2)]
#         # h = ops.sndeconv2d(h, new_shape, (c_h, 3), name='noise_expand', strides=(1, 2))
#         # tf.summary.histogram("before_blocks", h, family=scope_name)
#
#         h = tf.concat([compressed_reaction, h], axis=3, name="reaction_and_noise")
#         tf.summary.histogram("reaction_and_noise", h, family=scope_name)
#         # hidden_dim = gf_dim * (2 ** (number_of_layers - 1))
#         hidden_dim = h.get_shape().as_list()[3]
#         print("COMPRESSED TO: ", h.shape)
#
#         with tf.variable_scope("up", reuse=tf.AUTO_REUSE):
#             for layer_id in range(number_of_layers):
#                 block_name = 'up_block{}'.format(number_of_layers - (layer_id + 1))
#                 dilation_rate = (dilations[0] ** (number_of_layers - layer_id),
#                                  dilations[1] ** (number_of_layers - layer_id))
#                 h = sn_block_conditional(h, labels, hidden_dim, num_classes, block_name, get_kernel(h, kernel),
#                                          strides[layer_id], dilation_rate, tf.nn.leaky_relu, pooling, 'VALID')
#                 tf.summary.histogram(block_name, h, family=scope_name)
#                 if layer_id == number_of_layers - 2:
#                     h = attention(h, hidden_dim, sn=True, reuse=reuse)
#                     tf.summary.histogram("up_attention", h, family=scope_name)
#                 hidden_dim = hidden_dim / strides[layer_id][0]
#
#         bn = ops.BatchNorm(name='g_bn')
#         h_act = tf.nn.leaky_relu(bn(h), name="h_act")
#         out = tf.nn.tanh(ops.snconv2d(h_act, 1, (output_shape[0], 1), name='last_conv'), name="generated")
#         tf.summary.histogram("Generated_results", out, family=scope_name)
#         print("GENERATED SHAPE", out.shape)
#         return out


def generator_conditional(zs, labels, gf_dim, num_classes, kernel=(3, 3), strides=[(2, 2)], dilations=(1, 1),
                          pooling='avg', scope_name='Generator', reuse=False, output_shape=[8, 128, 1],
                          act=tf.nn.leaky_relu):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        labels = labels

        height_d, width_d = get_dimentions_factors(strides)
        number_of_layers = len(strides)
        hidden_dim = gf_dim * (2 ** (number_of_layers))
        number_of_layers = len(strides)

        c_h = int(output_shape[0] / height_d)
        c_w = int((output_shape[1] / width_d))
        h = ops.snlinear(zs, c_h * c_w * hidden_dim, name='noise_linear')
        h = tf.reshape(h, [-1, c_h, c_w, hidden_dim])
        print("COMPRESSED TO: ", h.shape)

        with tf.variable_scope("up", reuse=tf.AUTO_REUSE):
            for layer_id in range(number_of_layers):
                hidden_dim = hidden_dim / strides[layer_id][0]
                block_name = 'up_block{}'.format(number_of_layers - (layer_id + 1))
                dilation_rate = (dilations[0] ** (number_of_layers - layer_id),
                                 dilations[1] ** (number_of_layers - layer_id))
                h = sn_block_conditional(h, labels, hidden_dim, num_classes, block_name, (3,3),
                                         strides[layer_id], dilation_rate, tf.nn.leaky_relu, pooling, 'VALID')
                tf.summary.histogram(block_name, h, family=scope_name)
                if layer_id == number_of_layers - 2:
                    h = attention(h, hidden_dim, sn=True, reuse=reuse)
                    tf.summary.histogram("up_attention", h, family=scope_name)

        bn = ops.BatchNorm(name='g_bn')
        h_act = tf.nn.leaky_relu(bn(h), name="h_act")
        out = tf.nn.tanh(ops.snconv2d(h_act, 3, (3, 3), name='last_conv'), name="generated")
        tf.summary.histogram("Generated_results", out, family=scope_name)
        print("GENERATED SHAPE", out.shape)
        return out


def reaction_encoder(dilations, gf_dim, kernel, reactions, pooling, reuse, scope_name, input_width):
    compressed = []
    strides_schedule = [(2, 2), (2, 2), (2, 2)]
    down_layers = len(strides_schedule)
    with tf.variable_scope("down", reuse=tf.AUTO_REUSE):
        for i in range(4):
            hidden_dim = gf_dim / 8
            reaction_part = tf.transpose(
                ops.sn_embedding(tf.squeeze(reactions[:, i, :]), len(SMILES_CHARACTER_TO_ID), 8,
                                 name='component_embedding'), perm=[0, 2, 1])
            tf.summary.histogram("embedded_reactions", reaction_part, family=scope_name)
            h = tf.expand_dims(reaction_part, axis=3)
            if input_width < 256:
                h = tf.nn.leaky_relu(
                    ops.snconv2d(h, hidden_dim, (h.get_shape().as_list()[1], 3), name='first_component_conv',
                                 padding="CYCLE", strides=(1, 2)))
            tf.summary.histogram("first_component_conv", h, family=scope_name)
            for layer_id in range(down_layers):
                hidden_dim = hidden_dim * strides_schedule[layer_id][0]
                block_name = 'block{}'.format(layer_id)
                dilation_rate = dilations[0] ** (layer_id + 1), dilations[1] ** (layer_id + 1)
                h = ops.sn_norm_block(h, hidden_dim, block_name, get_kernel(h, kernel), strides_schedule[layer_id],
                                      dilation_rate, act=tf.nn.leaky_relu, pooling=pooling, padding="VALID")
                tf.summary.histogram(block_name, h, family=scope_name)
                if layer_id == 0:
                    h = attention(h, hidden_dim, sn=True, reuse=reuse)
                    tf.summary.histogram("attention", h, family=scope_name)
            compressed.append(h)
        print("Component Shape", compressed[0].shape)
        compressed_reaction = tf.concat([compressed[0], compressed[1], compressed[2], compressed[3]], axis=3,
                                        name="compressed_reaction")
        tf.summary.histogram("compressed_reaction", compressed_reaction, family=scope_name)
        print("REACTION COMPRESSED TO: ", compressed_reaction.shape)
    return compressed_reaction


def generator_rnn(zs, labels, gf_dim, num_classes, kernel=(3, 3), strides=(2, 2), dilations=(1, 1),
                  pooling='avg', scope_name='Generator', reuse=False, out_shape=[8, 128, 1], act=tf.nn.leaky_relu):
    num_layers = 3
    num_nodes = [gf_dim, gf_dim, int(out_shape[0] / 2)]
    with tf.variable_scope(scope_name, reuse=reuse):
        zs = tf.expand_dims(zs, axis=2)
        # Define LSTM cells
        enc_fw_cells = [LSTMCell(num_nodes[layer], name="fw_" + str(layer)) for layer in range(num_layers)]
        enc_bw_cells = [LSTMCell(num_nodes[layer], name="bw_" + str(layer)) for layer in range(num_layers)]

        # Connect LSTM cells bidirectionally and stack
        (all_states, fw_state, bw_state) = stack_bidirectional_dynamic_rnn(
            cells_fw=enc_fw_cells, cells_bw=enc_bw_cells, inputs=zs, dtype=tf.float32)

        output = tf.transpose(tf.squeeze(all_states), perm=[0, 2, 1])
        output = tf.nn.tanh(output, name="generated")
        print("GENERATED", all_states.shape)
    return output
