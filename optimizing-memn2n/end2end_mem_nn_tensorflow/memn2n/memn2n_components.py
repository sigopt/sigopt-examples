"""End-To-End Memory Networks.

The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""

import tensorflow as tf
import numpy as np
from six.moves import range


def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them 
    encoding[:, -1] = 1.0
    return np.transpose(encoding)

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.compat.v1.op_scope([t], name, "zero_nil_slot") as name:
        t = tf.convert_to_tensor(value=t, name="t")
        s = tf.shape(input=t)[1]
        z = tf.zeros(tf.stack([1, s]))
        return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.compat.v1.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(value=t, name="t")
        gn = tf.random.normal(tf.shape(input=t), stddev=stddev)
        return tf.add(t, gn, name=name)

class MemN2N(object):
    """End-To-End Memory Network."""
    def __init__(self, batch_size, vocab_size, sentence_size, memory_size, embedding_size, optimizer,
        hops=3,
        max_grad_norm=40.0,
        nonlin=None,
        initializer=tf.compat.v1.random_normal_initializer(stddev=0.1),
        encoding=position_encoding,
        session=tf.compat.v1.Session(),
        name='MemN2N'):
        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.

            optimizer = Tensorflow Optimizer to use to train network
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._name = name

        self._build_inputs()
        self._build_vars()

        self._optimizer = optimizer

        self._encoding = tf.constant(encoding(self._sentence_size, self._embedding_size), name="encoding")

        # cross entropy
        logits = self._inference(self._stories, self._queries) # (batch_size, vocab_size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.stop_gradient(tf.cast(self._answers, tf.float32)), name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(input_tensor=cross_entropy, name="cross_entropy_sum")

        # loss op
        loss_op = cross_entropy_sum

        # gradient pipeline
        grads_and_vars = self._optimizer.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._optimizer.apply_gradients(nil_grads_and_vars, name="train_op")

        # predict ops
        predict_op = tf.argmax(input=logits, axis=1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.math.log(predict_proba_op, name="predict_log_proba_op")

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op

        init_op = tf.compat.v1.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)

    def _build_inputs(self):
        self._stories = tf.compat.v1.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name="stories")
        self._queries = tf.compat.v1.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.compat.v1.placeholder(tf.int32, [None, self._vocab_size], name="answers")

    def _build_vars(self):
        with tf.compat.v1.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])
            C = tf.concat(axis=0, values=[ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])

            self.A_1 = tf.Variable(A, name="A")

            self.C = []

            for hopn in range(self._hops):
                with tf.compat.v1.variable_scope('hop_{}'.format(hopn)):
                    self.C.append(tf.Variable(C, name="C"))

        self._nil_vars = set([self.A_1.name] + [x.name for x in self.C])

    def _inference(self, stories, queries):
        with tf.compat.v1.variable_scope(self._name):
            # Use A_1 for the question embedding as per Adjacent Weight Sharing
            q_emb = tf.nn.embedding_lookup(params=self.A_1, ids=queries)
            u_0 = tf.reduce_sum(input_tensor=q_emb * self._encoding, axis=1)
            u = [u_0]

            for hopn in range(self._hops):
                if hopn == 0:
                    m_emb_A = tf.nn.embedding_lookup(params=self.A_1, ids=stories)
                    m_A = tf.reduce_sum(input_tensor=m_emb_A * self._encoding, axis=2)

                else:
                    with tf.compat.v1.variable_scope('hop_{}'.format(hopn - 1)):
                        m_emb_A = tf.nn.embedding_lookup(params=self.C[hopn - 1], ids=stories)
                        m_A = tf.reduce_sum(input_tensor=m_emb_A * self._encoding, axis=2)

                # hack to get around no reduce_dot
                u_temp = tf.transpose(a=tf.expand_dims(u[-1], -1), perm=[0, 2, 1])
                dotted = tf.reduce_sum(input_tensor=m_A * u_temp, axis=2)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)

                probs_temp = tf.transpose(a=tf.expand_dims(probs, -1), perm=[0, 2, 1])
                with tf.compat.v1.variable_scope('hop_{}'.format(hopn)):
                    m_emb_C = tf.nn.embedding_lookup(params=self.C[hopn], ids=stories)
                m_C = tf.reduce_sum(input_tensor=m_emb_C * self._encoding, axis=2)

                c_temp = tf.transpose(a=m_C, perm=[0, 2, 1])
                o_k = tf.reduce_sum(input_tensor=c_temp * probs_temp, axis=2)
                u_k = u[-1] + o_k

                # nonlinearity
                if self._nonlin:
                    u_k = nonlin(u_k)

                u.append(u_k)

            # Use last C for output (transposed)
            with tf.compat.v1.variable_scope('hop_{}'.format(self._hops)):
                return tf.matmul(u_k, tf.transpose(a=self.C[-1], perm=[1,0]))

    def batch_fit(self, stories, queries, answers):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers}
        loss, _ = self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, stories, queries):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_op, feed_dict=feed_dict)

    def predict_proba(self, stories, queries):
        """Predicts probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)

    def predict_log_proba(self, stories, queries):
        """Predicts log probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_log_proba_op, feed_dict=feed_dict)
