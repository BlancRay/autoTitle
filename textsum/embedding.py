# import urllib.request
import collections
import math
import os
import random
# import zipfile
import datetime as dt

import numpy as np
import tensorflow as tf

class embeded(object):
  def maybe_download(self,filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
      filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
      print('Found and verified', filename)
    else:
      print(statinfo.st_size)
      raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


  # Read the data into a list of strings.
  def read_data(self,filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
      data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

  def read_vocab(self,vocab_file):
    count = 0
    data = []
    with open(vocab_file, 'r') as vocab_f:
        for line in vocab_f:
          pieces = line.split()
          data.append(pieces[0])
          count += 1
    return data

  def build_dataset(self,words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
      dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
      if word in dictionary:
        index = dictionary[word]
      else:
        index = 0  # dictionary['UNK']
        unk_count += 1
      data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


  def collect_data(self,filename,vsize=10000):
    # url = 'http://mattmahoney.net/dc/'
    # filename = maybe_download('text8.zip', url, 31344016)
    # vocabulary = read_data(filename)
    # print(type(vocabulary))
    print("Reading vocab")
    # filename = "c:/Users/xulei/zhiziyun/autoTitle/Data/vocab_dic"
    vocabulary = self.read_vocab(filename)
    # vsize = len(vocabulary)
    self.vocabulary_size = len(vocabulary)
    print(type(vocabulary))
    print(vocabulary[:7])
    data, count, dictionary, reverse_dictionary = self.build_dataset(vocabulary,self.vocabulary_size)
    del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary



  # generate batch data
  def generate_batch(self,data, batch_size, num_skips, skip_window):
    # global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
      buffer.append(data[self.data_index])
      self.data_index = (self.data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
      target = skip_window  # input word at the center of the buffer
      targets_to_avoid = [skip_window]
      for j in range(num_skips):
        while target in targets_to_avoid:
          target = random.randint(0, span - 1)
        targets_to_avoid.append(target)
        # this is the input word
        batch[i * num_skips + j] = buffer[skip_window]
        # these are the context words
        context[i * num_skips + j, 0] = buffer[target]
      buffer.append(data[self.data_index])
      self.data_index = (self.data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    self.data_index = (self.data_index + len(data) - span) % len(data)
    return batch, context



  def run(self,graph, num_steps):
    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      self.init.run()
      print('Initialized')

      average_loss = 0
      for step in range(num_steps):
        batch_inputs, batch_context = self.generate_batch(self.data,self.batch_size, self.num_skips, self.skip_window)
        feed_dict = {self.train_inputs: batch_inputs,self.train_context: batch_context}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([self.optimizer, self.cross_entropy], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
          if step > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step ', step, ': ', average_loss)
          average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
          sim = self.similarity.eval()
          for i in range(self.valid_size):
            valid_word = self.reverse_dictionary[self.valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
              close_word = self.reverse_dictionary[nearest[k]]
              log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
      self.embedding = self.normalized_embeddings.eval()




  def get_embedding(self):
    self.data, self.count, self.dictionary, self.reverse_dictionary = self.collect_data(self.vocab_file,self.vocabulary_size)
    self.graph = tf.Graph()
    with self.graph.as_default():

      # Input data.
      self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
      self.train_context = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
      self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

      # Look up embeddings for inputs.
      embeddings = tf.Variable(
          tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
      embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)

      # Construct the variables for the softmax
      weights = tf.Variable(
          tf.truncated_normal([self.embedding_size, self.vocabulary_size],
                              stddev=1.0 / math.sqrt(self.embedding_size)))
      biases = tf.Variable(tf.zeros([self.vocabulary_size]))
      hidden_out = tf.transpose(
          tf.matmul(tf.transpose(weights), tf.transpose(embed))) + biases

      # convert train_context to a one-hot format
      train_one_hot = tf.one_hot(self.train_context, self.vocabulary_size)

      self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
          logits=hidden_out, labels=train_one_hot))

      # Construct the SGD optimizer using a learning rate of 1.0.
      self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.cross_entropy)

      # Compute the cosine similarity between minibatch examples and all embeddings.
      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
      self.normalized_embeddings = embeddings / norm
      valid_embeddings = tf.nn.embedding_lookup(
          self.normalized_embeddings, self.valid_dataset)
      self.similarity = tf.matmul(valid_embeddings, self.normalized_embeddings, transpose_b=True)

      # Add variable initializer.
      self.init = tf.global_variables_initializer()

    self.embedding = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
    softmax_start_time = dt.datetime.now()
    self.run(self.graph, num_steps=self.num_steps)
    softmax_end_time = dt.datetime.now()
    print("Softmax method took {} minutes to run 100 iterations".format((softmax_end_time-softmax_start_time).total_seconds()))

    # with nce_graph.as_default(self):

    #   # Construct the variables for the NCE loss
    #   nce_weights = tf.Variable(
    #       tf.truncated_normal([vocabulary_size, embedding_size],
    #                           stddev=1.0 / math.sqrt(embedding_size)))
    #   nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    #   nce_loss = tf.reduce_mean(
    #       tf.nn.nce_loss(weights=nce_weights,
    #                       biases=nce_biases,
    #                       labels=self.train_context,
    #                       inputs=embed,
    #                       num_sampled=num_sampled,
    #                       num_classes=vocabulary_size))

    #   self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(nce_loss)

    #   # Add variable initializer.
    #   init = tf.global_variables_initializer()


    # nce_start_time = dt.datetime.now()
    # run(graph, num_steps)
    # nce_end_time = dt.datetime.now()
    # print("NCE method took {} minutes to run 100 iterations".format((nce_end_time-nce_start_time).total_seconds()))

    return self.embedding

  def __init__(self,hps):
    self.vocab_file = hps.vocab_path
    self.data_index = 0
    self.vocabulary_size = 10000
    self.batch_size = hps.batch_size
    self.embedding_size = hps.emb_dim  # Dimension of the embedding vector.
    self.skip_window = 2       # How many words to consider left and right.
    self.num_skips = 2         # How many times to reuse an input to generate a label.
    self.num_steps = 50000
    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    self.valid_size = 16     # Random set of words to evaluate similarity on.
    self.valid_window = 100  # Only pick dev samples in the head of the distribution.
    self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
    self.num_sampled = 64    # Number of negative examples to sample.

