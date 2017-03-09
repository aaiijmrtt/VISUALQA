import sys, configparser, signal
import tensorflow as tf
import recentnet

def prepad(unpadded, pad, size):
	if len(unpadded) == size:
		return unpadded
	return [pad] * (size - len(unpadded)) + unpadded

def postpad(unpadded, pad, size):
	if len(unpadded) == size:
		return unpadded
	return unpadded + [pad] * (size - len(unpadded))

def pretrained(filename):
	return filename

def feed(model, config, filename):
	batch, length, align = config.getint('global', 'batchsize'), config.getint('global', 'timesize'), config.getboolean('global', 'align')
	images, questions, answers = list(), list(), list()
	for line in open(filename):
		filename, question, answer = line.split('\t')
		image = pretrained(filename)
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
		for ii, feeddict in enumerate(feed(model, config, filename)):
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
	tf.train.Saver().save(sess, config.get('global', 'save'))
	sys.exit()

if __name__ == '__main__':
	config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation())
	config.read(sys.argv[1])
	signal.signal(signal.SIGINT, handler)

	model = recentnet.create(config['recentnet'])
	with tf.Session() as sess:
		if sys.argv[3] == 'init':
			sess.run(tf.initialize_all_variables())
		else:
			tf.train.Saver().restore(sess, config.get('global', 'load'))
			summary = tf.train.SummaryWriter(config.get('global', 'logs'), sess.graph)
			print datetime.datetime.now(), 'running model'
			returnvalue = run(model, config, sess, summary, '%s/%s' %(config.get('global', 'data'), sys.argv[3]), sys.argv[3])
			print datetime.datetime.now(), 'returned value', returnvalue
		print datetime.datetime.now(), 'saving model'
		tf.train.Saver().save(sess, config.get('global', 'save'))
