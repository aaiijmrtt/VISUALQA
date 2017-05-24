import sys, configparser, signal, datetime
import tensorflow as tf, numpy as np, scipy.misc, scipy.ndimage
import tensorflow.contrib.slim.nets as nets
import pmnmtnet

def prepad(unpadded, pad, size):
	if len(unpadded) == size:
		return unpadded
	return [pad] * (size - len(unpadded)) + unpadded

def postpad(unpadded, pad, size):
	if len(unpadded) == size:
		return unpadded
	return unpadded + [pad] * (size - len(unpadded))

def featureextractor(placeholder):
	logits, subparts = nets.vgg.vgg_19(placeholder)
	return subparts['vgg_19/fc7']

def extractfeature(session, model, filename, proposals):
	image = scipy.misc.imresize(scipy.ndimage.imread(filename), [224, 224, 3])
	return [session.run(model['featureextractor'], feed_dict = {model['rawimage']: scipy.misc.imresize(image[proposal[0]: proposal[2], proposal[1]: proposal[3], :], [224, 224, 3]).reshape(1, 224, 224, 3)}) for proposal in proposals]

def feed(model, config, filename, session):
	batch, length, hidden, props = config.getint('global', 'batchsize'), config.getint('global', 'timesize'), config.getint('pmnmtnet', 'hidden'), config.getint('pmnmtnet', 'props')
	images, questions, answers = list(), list(), list()
	for line in open(filename):
		name, question, answer, proposals = line.split('\t')
		image = extractfeature(session, model, name, [[int(prop) for prop in proposal.split()] for proposal in proposals.split(':')])
		question = postpad([int(q) for q in question.split()], 0, length)
		answer = int(answer)
		images.append(postpad([im.reshape(4096) for im in image], np.zeros([4096], np.float32), props))
		questions.append(question)
		answers.append(answer)

		if len(questions) == batch:
			yield {model['query']: questions, model['image']: images, model['labels']: answers}
			images, questions, answers = list(), list(), list()

def run(model, config, session, summary, filename, train):
	iters, freq, time, saves, total = config.getint('global', 'iterations') if train == 'train' else 1, config.getint('global', 'frequency'), config.getint('global', 'timesize'), config.get('global', 'output'), 0.

	for i in xrange(iters):
		for ii, feeddict in enumerate(feed(model, config, filename, session)):
			if train == 'train':
				val, t = session.run([model['loss'], model['trn']], feed_dict = feeddict)
				total += val
				if (ii + 1) % freq == 0:
					print datetime.datetime.now(), 'iteration', i, 'batch', ii, 'loss:', val, total
			elif train == 'dev' or train == 'test':
				val = session.run(model['answer'], feed_dict = feeddict)
				exps, vals, outs = feeddict[model['labels']], val.values, val.indices
				for exp, out in zip(exps, outs):
					if exp in out: total += 1

	return total

def handler(signum, stack):
	print datetime.datetime.now(), 'terminating execution'
	print datetime.datetime.now(), 'saving model'
	tf.train.Saver(variables + [model[name] for name in newvariables]).save(sess, config.get('global', 'save'))
	sys.exit()

if __name__ == '__main__':
	config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
	config.read(sys.argv[1])
	signal.signal(signal.SIGINT, handler)

	with tf.Session() as sess:
		print datetime.datetime.now(), 'creating model'
		depth, model = config.getint('pmnmtnet', 'depth'), dict()
		model['rawimage'] = tf.placeholder(tf.float32, [1, 224, 224, 3])
		model['featureextractor'] = featureextractor(model['rawimage'])
		variables, newvariables = tf.contrib.slim.get_variables_to_restore(), ['wE', 'wT', 'bT'] + ['%s_%i' %(s, i) for s in ['w', 'b'] for i in xrange(depth)] + ['%s%s_%i' %(s1, s2, i) for s1 in ['w', 'b'] for s2 in ['Z', 'O'] for i in xrange(depth)] + ['%s%s%s_%i' %(s1, s2, s3, i) for s2 in ['f', 'b'] for s2 in ['w', 'b'] for s3 in ['C', 'F', 'I']]

		if sys.argv[2] == 'init':
			sess.run(tf.global_variables_initializer())
			tf.train.Saver(variables).restore(sess, config.get('global', 'preload'))
			model.update(pmnmtnet.create(config['pmnmtnet']))
			sess.run(tf.variables_initializer([model[name] for name in newvariables]))
			sess.run(tf.global_variables_initializer())
		else:
			model.update(pmnmtnet.create(config['pmnmtnet']))
			sess.run(tf.global_variables_initializer())
			tf.train.Saver(variables + [model[name] for name in newvariables]).restore(sess, config.get('global', 'load'))
			print datetime.datetime.now(), 'running model'
			summary = tf.summary.FileWriter(config.get('global', 'logs'), sess.graph)
			returnvalue = run(model, config, sess, summary, '%s/%s' %(config.get('global', 'data'), sys.argv[2]), sys.argv[2])
			print datetime.datetime.now(), 'returned value', returnvalue

		print datetime.datetime.now(), 'saving model'
		tf.train.Saver(variables + [model[name] for name in newvariables]).save(sess, config.get('global', 'save'))
