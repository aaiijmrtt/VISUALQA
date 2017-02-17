import tensorflow as tf, numpy as np

def create(config):
	dim_b, dim_ih, dim_iw, dim_ic, dim_qv, dim_qi, dim_qt, dim_o = config.getint('batch'), config.getint('height'), config.getint('width'), config.getint('channels'), config.getint('vocab'), config.getint('embed'), config.getint('time'), config.getint('answers')
	nlinear, preds, lrate, dstep, drate, optim, rfact, reg = getattr(tf.nn, config.get('nlinear')), config.getint('preds'), config.getfloat('lrate'), config.getint('dstep'), config.getfloat('drate'), getattr(tf.train, config.get('optim')), config.getfloat('rfact'), getattr(tf.contrib.layers, config.get('reg'))

	model = dict()

	with tf.name_scope('embedding'):
		model['wE'] = tf.Variable(tf.random_uniform([dim_qi, dim_qi], - np.sqrt(6. / (dim_qi + dim_qi)), np.sqrt(6. / (dim_qi + dim_qi))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'wE')

	with tf.name_scope('input_weights'):
		model['Fq'] = tf.Variable(tf.random_uniform([dim_qt, dim_qi], - np.sqrt(6. / (dim_qt + dim_qi)), np.sqrt(6. / (dim_qt + dim_qi))), collections = [tf.GraphKeys.GLOBAL_VARIABLES], name = 'Fq')
		model['Fv'] = tf.Variable(tf.random_uniform([dim_iw, dim_ic], - np.sqrt(6. / (dim_iw + dim_ic)), np.sqrt(6. / (dim_iw + dim_ic))), collections = [tf.GraphKeys.GLOBAL_VARIABLES], name = 'Fv')

	with tf.name_scope('query'):
		model['query'] = tf.placeholder(tf.int32, [dim_b, dim_qt], name = 'query')
		model['q'] = tf.reduce_sum(tf.multiply(tf.nn.embedding_lookup(model['wE'], model['query']), model['Fq']), 1, name = 'q')

	with tf.name_scope('image'):
		model['image'] = tf.placeholder(tf.float32, [dim_b, dim_ih, dim_iw, dim_ic], name = 'image')
		model['im'] = tf.transpose(tf.reshape(model['image'], [dim_b, dim_ih * dim_iw, dim_ic]), [1, 0, 2], name = 'im')
		model['i'] = tf.unstack(tf.transpose(tf.reduce_sum(tf.multiply(model['image'], model['Fv']), 2), [1, 0, 2]), name = 'i')

	with tf.name_scope('initial'):
		model['h_'] = tf.Variable(tf.random_uniform([dim_ih * dim_iw, dim_ic], - np.sqrt(6. / (dim_ih * dim_iw + dim_ic)), np.sqrt(6. / (dim_ih * dim_iw + dim_ic))), collections = [tf.GraphKeys.GLOBAL_VARIABLES], name = 'h_')
		model['h_-1'] = tf.stack([model['h_'] for i in xrange(dim_b)], name = 'h_-1')

	with tf.name_scope('transform_weights'):
		model['wU'] = tf.Variable(tf.random_uniform([dim_ic, dim_ic], - np.sqrt(6. / (dim_ic + dim_ic)), np.sqrt(6. / (dim_ic + dim_ic))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'wU')
		model['wV'] = tf.Variable(tf.random_uniform([dim_ic, dim_ic], - np.sqrt(6. / (dim_ic + dim_ic)), np.sqrt(6. / (dim_ic + dim_ic))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'wV')
		model['wW'] = tf.Variable(tf.random_uniform([dim_ic, dim_ic], - np.sqrt(6. / (dim_ic + dim_ic)), np.sqrt(6. / (dim_ic + dim_ic))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'wW')
		model['wH'] = tf.Variable(tf.random_uniform([dim_ic, dim_ic], - np.sqrt(6. / (dim_ic + dim_ic)), np.sqrt(6. / (dim_ic + dim_ic))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'wH')
		model['wR'] = tf.Variable(tf.random_uniform([dim_ic, dim_o], - np.sqrt(6. / (dim_ic + dim_o)), np.sqrt(6. / (dim_ic + dim_o))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'wR')

	for i in xrange(dim_ih):
		with tf.name_scope('gate_%i' %i):
			model['gl_%i' %i] = tf.reduce_sum(tf.multiply(model['i'][i], model['im']), 2, name = 'gl_%i' %i)
			model['gr_%i' %i] = tf.stack([tf.reduce_sum(tf.multiply(model['i'][i], h), 1) for h in tf.unstack(tf.transpose(model['h_%i' %(i - 1)], [1, 0, 2]))], name = 'gr_%i' %i)
			model['g_%i' %i] = tf.expand_dims(tf.transpose(tf.nn.sigmoid(tf.add(model['gl_%i' %i], model['gr_%i' %i])), [1, 0]), 2, name = 'g_%i' %i)

		with tf.name_scope('state_%i' %i):
			model['hl_%i' %i] = tf.transpose(tf.stack([tf.matmul(h, model['wU']) for h in tf.unstack(tf.transpose(model['h_%i' %(i - 1)], [1, 0, 2]))]), [1, 0, 2], name = 'hl_%i' %i)
			model['hc_%i' %i] = tf.stack([tf.matmul(im, model['wV']) for im in tf.unstack(tf.transpose(model['im'], [1, 0, 2]))], name = 'hc_%i' %i)
			model['hr_%i' %i] = tf.expand_dims(tf.matmul(model['i'][i], model['wW']), 1, name = 'hr_%i' %i)
			model['hbar_%i' %i] = nlinear(tf.add(tf.add(model['hl_%i' %i], model['hc_%i' %i]), model['hr_%i' %i]), 'hbar_%i' %i)
			model['h_%i' %i] = tf.nn.l2_normalize(tf.add(model['h_%i' %(i - 1)], tf.multiply(model['g_%i' %i], model['hbar_%i' %i])), 2, name = 'h_%i' %i)

	with tf.name_scope('response'):
		model['h'] = model['h_%i' %(dim_ih - 1)]
		model['p'] = tf.nn.softmax(tf.reduce_sum(tf.multiply(model['h'], tf.expand_dims(model['q'], 1)), 2), name = 'p')
		model['u'] = tf.reduce_sum(tf.multiply(tf.expand_dims(model['p'], 2), model['h']), 1, name = 'u')
		model['y'] = tf.matmul(tf.nn.tanh(tf.add(model['q'], tf.matmul(model['u'], model['wH']))), model['wR'], name = 'y')

	with tf.name_scope('result'):
		model['labels'] = tf.placeholder(tf.int32, [dim_b], name = 'labels')
		model['answer'] = tf.nn.top_k(model['y'], preds, name = 'answer')
		model['loss'] = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = model['labels'], logits = model['y'], name = 'loss')

	model['gs'] = tf.Variable(0, trainable = False, name = 'gs')
	model['lr'] = tf.train.exponential_decay(lrate, model['gs'], dstep, drate, staircase = False, name = 'lr')
	model['reg'] = tf.contrib.layers.apply_regularization(reg(rfact), tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
	model['trn'] = optim(model['lr']).minimize(model['loss'] + model['reg'], global_step = model['gs'], name = 'trn')

	return model
