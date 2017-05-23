import tensorflow as tf, numpy as np

def create(config):
	dim_b, dim_ip, dim_ih, dim_qv, dim_qi, dim_qt, dim_d, dim_o = config.getint('batch'), config.getint('props'), config.getint('hidden'), config.getint('vocab'), config.getint('embed'), config.getint('time'), config.getint('depth'), config.getint('answers')
	nlinear, preds, lrate, dstep, drate, optim, rfact, reg = getattr(tf.nn, config.get('nlinear')), config.getint('preds'), config.getfloat('lrate'), config.getint('dstep'), config.getfloat('drate'), getattr(tf.train, config.get('optim')), config.getfloat('rfact'), getattr(tf.contrib.layers, config.get('reg'))

	assert dim_ih == dim_qi, 'dim_ih[%i] != dim_qi[%i]' %(dim_ih, dim_qi)
	model = dict()

	with tf.name_scope('embedding'):
		model['wE'] = tf.Variable(tf.random_uniform([dim_qv, dim_qi], - np.sqrt(6. / (dim_qi + dim_qi)), np.sqrt(6. / (dim_qi + dim_qi))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'wE')

	with tf.name_scope('query'):
		model['query'] = tf.placeholder(tf.int32, [dim_b, dim_qt], name = 'query')
		model['qr'] = tf.transpose(model['query'], [1, 0], name = 'qr')
		model['qt'] = tf.unstack(model['qr'], name = 'qt')
		model['q'] = tf.unstack(tf.nn.embedding_lookup(model['wE'], model['qr']), name = 'q')

	with tf.name_scope('image'):
		model['image'] = tf.placeholder(tf.float32, [dim_b, dim_ip, 4096], name = 'image')
		model['im'] = tf.unstack(tf.transpose(model['image'], [1, 0, 2]), name = 'im')
		model['wT'] = tf.Variable(tf.random_uniform([4096, dim_ih], - np.sqrt(6. / (4096 + dim_ih)), np.sqrt(6. / (4096 + dim_ih))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'wT')
		model['bT'] = tf.Variable(tf.random_uniform([1, dim_ih], - np.sqrt(6. / dim_ih), np.sqrt(6. / dim_ih)), name = 'bT')
		for i in xrange(dim_d):
			model['c_%i_%i' %(i, -1)] = [nlinear(tf.add(tf.matmul(im, model['wT']), model['bT'])) for im in model['im']]
			model['c_%i_%i' %(i, dim_qt)] = [nlinear(tf.add(tf.matmul(im, model['wT']), model['bT'])) for im in model['im']]

	for i in xrange(dim_d):
		with tf.name_scope('input_%i' %i):
			for ii in xrange(dim_qt):
				for iii in xrange(dim_ip):
					model['x_%i_%i_%i' %(i, ii, iii)] = model['q'][i] if i == 0 else model['h_%i_%i_%i' %(i - 1, ii, iii)]

		with tf.name_scope('inputgate_%i' %i):
			model['fwI_%i' %i] = tf.Variable(tf.random_uniform([dim_qi, dim_qi], - np.sqrt(6. / (dim_qi + dim_qi)), np.sqrt(6. / (dim_qi + dim_qi))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'fwI_%i' %i)
			model['fbI_%i' %i] = tf.Variable(tf.random_uniform([1, dim_qi], - np.sqrt(6. / dim_qi), np.sqrt(6. / dim_qi)), name = 'fbI_%i' %i)
			model['bwI_%i' %i] = tf.Variable(tf.random_uniform([dim_qi, dim_qi], - np.sqrt(6. / (dim_qi + dim_qi)), np.sqrt(6. / (dim_qi + dim_qi))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'bwI_%i' %i)
			model['bbI_%i' %i] = tf.Variable(tf.random_uniform([1, dim_qi], - np.sqrt(6. / dim_qi), np.sqrt(6. / dim_qi)), name = 'bbI_%i' %i)
			for ii in xrange(dim_qt):
				for iii in xrange(dim_ip):
					model['fi_%i_%i_%i' %(i, ii, iii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['x_%i_%i_%i' %(i, ii, iii)], model['fwI_%i' %i]), model['fbI_%i' %i]), name = 'fi_%i_%i_%i' %(i, ii, iii))
					model['bi_%i_%i_%i' %(i, ii, iii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['x_%i_%i_%i' %(i, ii, iii)], model['bwI_%i' %i]), model['bbI_%i' %i]), name = 'bi_%i_%i_%i' %(i, ii, iii))

		with tf.name_scope('forgetgate_%i' %i):
			model['fwF_%i' %i] = tf.Variable(tf.random_uniform([dim_qi, dim_qi], - np.sqrt(6. / (dim_qi + dim_qi)), np.sqrt(6. / (dim_qi + dim_qi))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'fwF_%i' %i)
			model['fbF_%i' %i] = tf.Variable(tf.random_uniform([1, dim_qi], - np.sqrt(6. / dim_qi), np.sqrt(6. / dim_qi)), name = 'fbF_%i' %i)
			model['bwF_%i' %i] = tf.Variable(tf.random_uniform([dim_qi, dim_qi], - np.sqrt(6. / (dim_qi + dim_qi)), np.sqrt(6. / (dim_qi + dim_qi))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'bwF_%i' %i)
			model['bbF_%i' %i] = tf.Variable(tf.random_uniform([1, dim_qi], - np.sqrt(6. / dim_qi), np.sqrt(6. / dim_qi)), name = 'bbF_%i' %i)
			for ii in xrange(dim_qt):
				for iii in xrange(dim_ip):
					model['ff_%i_%i_%i' %(i, ii, iii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['x_%i_%i_%i' %(i, ii, iii)], model['fwF_%i' %i]), model['fbF_%i' %i]), name = 'fF_%i_%i_%i' %(i, ii, iii))
					model['bf_%i_%i_%i' %(i, ii, iii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['x_%i_%i_%i' %(i, ii, iii)], model['bwF_%i' %i]), model['bbF_%i' %i]), name = 'bF_%i_%i_%i' %(i, ii, iii))

		with tf.name_scope('outputgate_%i' %i):
			model['wO_%i' %i] = tf.Variable(tf.random_uniform([dim_qi, dim_qi], - np.sqrt(6. / (dim_qi + dim_qi)), np.sqrt(6. / (dim_qi + dim_qi))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'wO_%i' %i)
			model['bO_%i' %i] = tf.Variable(tf.random_uniform([1, dim_qi], - np.sqrt(6. / dim_qi), np.sqrt(6. / dim_qi)), name = 'bO_%i' %i)
			for ii in xrange(dim_qt):
				for iii in xrange(dim_ip):
					model['o_%i_%i_%i' %(i, ii, iii)] = tf.nn.sigmoid(tf.add(tf.matmul(model['x_%i_%i_%i' %(i, ii, iii)], model['wO_%i' %i]), model['bO_%i' %i]), name = 'o_%i_%i_%i' %(i, ii, iii))

		with tf.name_scope('cellstate_%i' %i):
			model['fwC_%i' %i] = tf.Variable(tf.random_uniform([dim_qi, dim_qi], - np.sqrt(6. / (dim_qi + dim_qi)), np.sqrt(6. / (dim_qi + dim_qi))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'fwC_%i' %i)
			model['fbC_%i' %i] = tf.Variable(tf.random_uniform([1, dim_qi], - np.sqrt(6. / dim_qi), np.sqrt(6. / dim_qi)), name = 'fbC_%i' %i)
			model['bwC_%i' %i] = tf.Variable(tf.random_uniform([dim_qi, dim_qi], - np.sqrt(6. / (dim_qi + dim_qi)), np.sqrt(6. / (dim_qi + dim_qi))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'bwC_%i' %i)
			model['bbC_%i' %i] = tf.Variable(tf.random_uniform([1, dim_qi], - np.sqrt(6. / dim_qi), np.sqrt(6. / dim_qi)), name = 'bbC_%i' %i)
			for ii in xrange(dim_qt):
				for iii in xrange(dim_ip):
					model['fcc_%i_%i_%i' %(i, ii, iii)] = model['c_%i_%i' %(i, ii - 1)][iii] if ii == 0 else model['fc_%i_%i_%i' %(i, ii - 1, iii)]
					model['fc_%i_%i_%i' %(i, ii, iii)] = tf.where(tf.equal(model['qt'][ii], tf.zeros([dim_b], tf.int32)), model['fcc_%i_%i_%i' %(i, ii, iii)], tf.add(tf.multiply(model['ff_%i_%i_%i' %(i, ii, iii)], model['fcc_%i_%i_%i' %(i, ii, iii)]), tf.multiply(model['fi_%i_%i_%i' %(i, ii, iii)], tf.nn.tanh(tf.add(tf.matmul(model['x_%i_%i_%i' %(i, ii, iii)], model['fwC_%i' %i]), model['fbC_%i' %i])))), name = 'fc_%i_%i_%i' %(i, ii, iii))

			for ii in reversed(xrange(dim_qt)):
				for iii in xrange(dim_ip):
					model['bcc_%i_%i_%i' %(i, ii, iii)] = model['c_%i_%i' %(i, ii + 1)][iii] if ii == dim_qt - 1 else model['bc_%i_%i_%i' %(i, ii + 1, iii)]
					model['bc_%i_%i_%i' %(i, ii, iii)] = tf.where(tf.equal(model['qt'][ii], tf.zeros([dim_b], tf.int32)), model['bcc_%i_%i_%i' %(i, ii, iii)], tf.add(tf.multiply(model['bf_%i_%i_%i' %(i, ii, iii)], model['bcc_%i_%i_%i' %(i, ii, iii)]), tf.multiply(model['bi_%i_%i_%i' %(i, ii, iii)], tf.nn.tanh(tf.add(tf.matmul(model['x_%i_%i_%i' %(i, ii, iii)], model['bwC_%i' %i]), model['bbC_%i' %i])))), name = 'bc_%i_%i_%i' %(i, ii, iii))

			for ii in xrange(dim_qt):
				for iii in xrange(dim_ip):
					model['c_%i_%i_%i' %(i, ii, iii)] = tf.concat([model['fc_%i_%i_%i' %(i, ii, iii)], model['bc_%i_%i_%i' %(i, ii, iii)]], 1, 'c_%i_%i_%i' %(i, ii, iii))

		with tf.name_scope('hidden_%i' %i):
			model['wZ_%i' %i] = tf.Variable(tf.random_uniform([2 * dim_qi, dim_qi], - np.sqrt(6. / (dim_qi + dim_qi)), np.sqrt(6. / (dim_qi + dim_qi))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'wZ_%i' %i)
			model['bZ_%i' %i] = tf.Variable(tf.random_uniform([1, dim_qi], - np.sqrt(6. / dim_qi), np.sqrt(6. / dim_qi)), name = 'bZ_%i' %i)
			for ii in xrange(dim_qt):
				for iii in xrange(dim_ip):
					model['z_%i_%i_%i' %(i, ii, iii)] = tf.add(tf.matmul(model['c_%i_%i_%i' %(i, ii, iii)], model['wZ_%i' %i]), model['bZ_%i' %i], name = 'z_%i_%i_%i' %(i, ii, iii))

		with tf.name_scope('output_%i' %i):
			for ii in xrange(dim_qt):
				for iii in xrange(dim_ip):
					model['h_%i_%i_%i' %(i, ii, iii)] = tf.multiply(model['o_%i_%i_%i' %(i, ii, iii)], tf.nn.tanh(model['z_%i_%i_%i' %(i, ii, iii)]), name = 'h_%i_%i_%i' %(i, ii, iii))

	with tf.name_scope('response'):
		model['h'] = tf.transpose(tf.stack([model['h_%i_%i_%i' %(dim_d - 1, dim_qt - 1, i)] for i in xrange(dim_ip)]), [1, 0, 2], name = 'h')
		model['x'] = tf.reduce_sum(tf.where(tf.transpose(tf.reshape(tf.tile(tf.equal(tf.reduce_sum(model['image'], 2), tf.zeros([dim_b, dim_ip], tf.float32)), [1, dim_ih]), [dim_b, dim_ih, dim_ip]), [0, 2, 1]), tf.zeros([dim_b, dim_ip, dim_ih], tf.float32), model['h']), 1, name = 'y')

	for i in xrange(dim_d):
		with tf.name_scope('layer_%i' %i):
			model['w_%i' %i] = tf.Variable(tf.random_uniform([dim_qi, dim_o], - np.sqrt(6. / (dim_qi + dim_o)), np.sqrt(6. / (dim_qi + dim_o))), name = 'w_%i' %i) if i == dim_d - 1 else tf.Variable(tf.random_uniform([dim_qi, dim_qi], - np.sqrt(6. / (dim_qi + dim_qi)), np.sqrt(6. / (dim_qi + dim_qi))), collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES], name = 'w_%i' %i)
			model['b_%i' %i] = tf.Variable(tf.random_uniform([1, dim_o], - np.sqrt(6. / dim_o), np.sqrt(6. / dim_o)), name = 'b_%i' %i) if i == dim_d - 1 else tf.Variable(tf.random_uniform([1, dim_qi], - np.sqrt(6. / dim_qi), np.sqrt(6. / dim_qi)), name = 'b_%i' %i)
			model['x_%i' %i] = model['x'] if i == 0 else nlinear(model['y_%i' %(i - 1)], name = 'x_%i' %i)
			model['y_%i' %i] = tf.add(tf.matmul(model['x_%i' %i], model['w_%i' %i]), model['b_%i' %i], name = 'y_%i' %i)


	with tf.name_scope('result'):
		model['output'] = model['y_%i' %(dim_d - 1)]
		model['labels'] = tf.placeholder(tf.int32, [dim_b], name = 'labels')
		model['answer'] = tf.nn.top_k(model['output'], preds, name = 'answer')
		model['loss'] = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = model['labels'], logits = model['output'], name = 'loss')

	model['gs'] = tf.Variable(0, trainable = False, name = 'gs')
	model['lr'] = tf.train.exponential_decay(lrate, model['gs'], dstep, drate, staircase = False, name = 'lr')
	model['reg'] = tf.contrib.layers.apply_regularization(reg(rfact), tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
	model['trn'] = optim(model['lr']).minimize(model['loss'] + model['reg'], global_step = model['gs'], name = 'trn')

	return model
