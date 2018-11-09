import tensorflow as tf
ACID_EMBEDDINGS = "acid_embeddings"
ACID_EMBEDDINGS_SCOPE = "acid_emb_scope"
REAL_PROTEINS = "real_proteins"
FAKE_PROTEINS = "fake_proteins"


def convert_to_acid_ids(fake_x, batch_size):
    fake_to_display = tf.squeeze(fake_x)
    acid_embeddings = tf.get_variable(ACID_EMBEDDINGS_SCOPE + "/" + ACID_EMBEDDINGS)
    fake_to_display = reverse_embedding_lookup(acid_embeddings, fake_to_display, batch_size)
    fake_to_display = tf.squeeze(fake_to_display, name=FAKE_PROTEINS)
    return fake_to_display

def reverse_embedding_lookup(acid_embeddings, embedded_sequence, batch_size):
    embedded_sequence = tf.transpose(embedded_sequence, perm=[0, 2, 1]) + tf.constant(10.0)
    acid_embeddings = acid_embeddings + tf.constant(10.0)
    acid_embeddings_expanded = tf.tile(tf.expand_dims(acid_embeddings, axis=0), [batch_size, 1, 1])
    emb_distances = tf.matmul(
        tf.nn.l2_normalize(acid_embeddings_expanded, axis=2),
        tf.nn.l2_normalize(embedded_sequence, axis=2),
        transpose_b=True)
    indices = tf.argmax(emb_distances, axis=1)
    return indices

def test_amino_acid_embeddings(acid_embeddings, real_x, width):
    real_x = tf.Print(real_x, [tf.transpose(real_x[0], perm=[1, 0])[127, :]], "REAL_127",
                      summarize=width)
    real_x = tf.Print(real_x, [tf.transpose(real_x[0], perm=[1, 0])[0, :]], "REAL_0",
                      summarize=width)
    real_x = reverse_embedding_lookup(acid_embeddings, real_x)
    real_x = tf.Print(real_x, [real_x[0]], "RECO!: ", summarize=width)
    tf.summary.histogram("test", real_x[0], family="test_reconstruction")
