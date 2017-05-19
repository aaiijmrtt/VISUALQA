import sys, configparser, signal, datetime
import tensorflow as tf, numpy as np, scipy.misc
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
	return subparts['vgg_19/conv5/conv5_4']

def extractfeature(session, model, filename, proposals):
	image = np.load(filename).reshape(224, 224, 3)
	proposals = [scipy.misc.imresize(image[proposal[0]: proposal[2], proposal[1]: proposal[3], :], [224, 224, 3]) for proposal in proposals]
	return session.run(model['featureextractor'], feed_dict = {model['rawimage']: proposals})

def feed(model, config, filename, session):
	batch, length = config.getint('global', 'batchsize'), config.getint('global', 'timesize')
	images, questions, answers = list(), list(), list()
	for line in open(filename):
		name, question, answer, proposals = line.split('\t')
		image = extractfeature(session, model, name, [[int(prop) for prop in proposal.split()] for proposal in proposals])
		question = postpad([int(q) for q in question.split()], 0, length)
		answer = int(answer)
		images.append(image)
		questions.append(question)
		answers.append(answer)

		if len(questions) == batch:
			yield {model['query']: questions, model['image']: images, model['labels']: answers}
			images, questions, answers = list(), list(), list()

def run(model, config, session, summary, filename, train):
	iters, freq, time, saves, total = config.getint('global', 'iterations') if train else 1, config.getint('global', 'frequency'), config.getint('global', 'timesize'), config.get('global', 'output'), 0.

	for i in xrange(iters):
		for ii, feeddict in enumerate(feed(model, config, filename, session)):
			if train == 'train':
				val, t = session.run([model['loss'], model['trn']], feed_dict = feeddict)
				total += val
				if (ii + 1) % freq == 0:
					print datetime.datetime.now(), 'iteration', i, 'batch', ii, 'loss:', val, total
			elif train == 'dev' or train == 'test':
				val = session.run(model['answer'], feed_dict = feeddict)
				exps, vals, outs = feeddict[model['labels']], [x[0] for x in val], [x[1] for x in val]
				for exp, out in zip(exps, outs):
					if exp == out: total += 1

	return total

def handler(signum, stack):
	print datetime.datetime.now(), 'terminating execution'
	print datetime.datetime.now(), 'saving model'
	tf.train.Saver(variables + newvariables).save(sess, config.get('global', 'save'))
	sys.exit()

if __name__ == '__main__':
	config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
	config.read(sys.argv[1])
	signal.signal(signal.SIGINT, handler)

	with tf.Session() as sess:
		print datetime.datetime.now(), 'creating model'
		model = dict()
		model['rawimage'] = tf.placeholder(tf.float32, [1, 224, 224, 3])
		model['featureextractor'] = featureextractor(model['rawimage'])
		variables, newvariables = tf.contrib.slim.get_variables_to_restore(), [model[name] for name in ['wE', 'Fq', 'Fv', 'h_', 'wU', 'wV', 'wW', 'wH', 'wR', 'gs']]

		if sys.argv[2] == 'init':
			sess.run(tf.global_variables_initializer())
			tf.train.Saver(variables).restore(sess, config.get('global', 'preload'))
			model.update(recentnet.create(config['pmnmtnet']))
			sess.run(tf.initialize_variables(newvariables))
		else:
			model.update(recentnet.create(config['recentnet']))
			sess.run(tf.initialize_all_variables())
			tf.train.Saver(variables + newvariables).restore(sess, config.get('global', 'load'))
			print datetime.datetime.now(), 'running model'
			summary = tf.summary.FileWriter(config.get('global', 'logs'), sess.graph)
			returnvalue = run(model, config, sess, summary, '%s/%s' %(config.get('global', 'data'), sys.argv[2]), sys.argv[2])
			print datetime.datetime.now(), 'returned value', returnvalue

		print datetime.datetime.now(), 'saving model'
		tf.train.Saver(variables + newvariables).save(sess, config.get('global', 'save'))
