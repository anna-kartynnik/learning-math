import argparse
import random
import time
import traceback

import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim

from preprocess_dataset.data_processor import DataProcessor, prepare_folds, read_files
from utils.gpu_utils import get_available_device, is_gpu_available
from models import Graph2TreeModel
from utils.calculate import compute_equations_result
from preprocess_dataset.batch import MATHBatch


class Trainer(object):
	"""Trains the model."""

	def __init__(self, args, samples):
		super(Trainer, self).__init__()
		self.args = args
		self.samples = samples

		if self.args.seed is not None:
			self.set_random_seed(self.args.seed)

		self.device = get_available_device(verbose=True)

		self.data_processor = DataProcessor(self.samples, trim_min_count=self.args.min_freq)

	def initialize(self, train_pairs, test_pairs):
		# Prepare data, get input/output languages
		self.data_processor.initialize(train_pairs, test_pairs)

		self.build_model()
		self.build_optimizer()

	def build_model(self):
		op_nums = self.data_processor.output_lang.n_words - self.data_processor.copy_nums - 1 - len(self.data_processor.generate_nums)

		self.model = Graph2TreeModel(
			self.args.embedding_size,
			self.args.hidden_size,
			op_nums,
			len(self.data_processor.generate_nums) + len(self.data_processor.var_nums),
			self.data_processor.var_nums,
			self.args.freeze_emb,
		)
		self.model.to(self.device)

	def build_optimizer(self):
		self.embedding_optimizer = optim.Adam(self.model.embedding.parameters(), lr=self.args.emb_learning_rate, weight_decay=self.args.weight_decay)
		self.encoder_optimizer = optim.Adam(self.model.encoder.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
		self.predictor_optimizer = optim.Adam(self.model.predictor.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
		self.generator_optimizer = optim.Adam(self.model.generator.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
		self.merger_optimizer = optim.Adam(self.model.merger.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
		self.sem_align_optimizer = optim.Adam(self.model.semantic_alignment.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

		self.embedding_scheduler = optim.lr_scheduler.StepLR(self.embedding_optimizer, step_size=20, gamma=0.5)
		self.encoder_scheduler = optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=20, gamma=0.5)
		self.predictor_scheduler = optim.lr_scheduler.StepLR(self.predictor_optimizer, step_size=20, gamma=0.5)
		self.generator_scheduler = optim.lr_scheduler.StepLR(self.generator_optimizer, step_size=20, gamma=0.5)
		self.merger_scheduler = optim.lr_scheduler.StepLR(self.merger_optimizer, step_size=20, gamma=0.5)
		self.sem_align_scheduler = optim.lr_scheduler.StepLR(self.sem_align_optimizer, step_size=20, gamma=0.5)

	def set_random_seed(self, seed):
		"""Sets seed for reproducibility."""
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)

	def scheduler_step(self):
		self.embedding_scheduler.step()
		self.encoder_scheduler.step()
		self.predictor_scheduler.step()
		self.generator_scheduler.step()
		self.merger_scheduler.step()
		self.sem_align_optimizer.step()

	def optimizer_zero_grad(self):
		self.embedding_optimizer.zero_grad()
		self.encoder_optimizer.zero_grad()
		self.predictor_optimizer.zero_grad()
		self.generator_optimizer.zero_grad()
		self.merger_optimizer.zero_grad()
		self.sem_align_optimizer.zero_grad()

	def optimizer_step(self):
		self.embedding_optimizer.step()
		self.encoder_optimizer.step()
		self.predictor_optimizer.step()
		self.generator_optimizer.step()
		self.merger_optimizer.step()
		self.sem_align_optimizer.step()

	def train(self):
		best_val_acc = -1

		print('---------------- Checking validation:')
		self.eval(mode='val-check')

		print('------------------\nStarting training.')
		for epoch in range(1, self.args.max_epochs + 1):
			self.model.train()
			loss_to_print, errors = self.train_epoch(epoch, errors=0)
			print('epochs = {}, train_loss = {:.3f}, errors = {}'.format(epoch, loss_to_print, errors))

			if epoch > 0 and epoch % 5 == 0:
				val_acc = self.eval(mode='val')
				if val_acc > best_val_acc:
					best_val_acc = val_acc
		print('Best validation accuracy: {:.3f}\n'.format(best_val_acc))
		return best_val_acc

	def train_epoch(self, epoch, errors):
		loss_to_print = 0
		batches = self.data_processor.prepare_batches(self.data_processor.train_pairs, self.args.batch_size)
		number_of_batches = len(batches)

		for step, batch in tqdm(enumerate(batches), desc=f'Epoch {epoch:02d}', total=number_of_batches):
			self.optimizer_zero_grad()

			try:
				loss = self.model(batch, self.data_processor.generate_num_ids, self.data_processor.input_lang,
							self.data_processor.output_lang, var_nums=self.data_processor.var_nums)
				loss.backward()
				self.optimizer_step()
				loss_to_print += loss
			except Exception as e:
				print(e)
				traceback.print_exc()
				errors += 1

		self.scheduler_step()

		return loss_to_print / number_of_batches, errors

	def canonical(self, value, digits=2):
		try:
			return round(float(value), ndigits=digits)
		except ValueError:
			# Conversion to `float` failed.
			return np.nan

	def eval(self, mode='val', size=None):
		self.model.eval()
		start = time.time()
		value_ac = 0
		equation_ac = 0
		answer_ac = 0
		eval_total = 0
		correct = 0
		skipped = 0

		if mode == 'val-check':
			test_batch = self.data_processor.test_pairs[:10]
		else:
			test_batch = self.data_processor.test_pairs

		for step, test_item in tqdm(enumerate(test_batch), desc='Validation', total=len(test_batch)):
			if test_item.filename == 'augmented':
				continue
			target_ans = test_item.answers
			try:
				test_ans = 'undefined'
				test_res = self.model.evaluate_tree(test_item, self.data_processor.generate_num_ids, self.data_processor.input_lang,
						self.data_processor.output_lang, beam_size=self.args.beam_size, beam_search=self.args.beam_search,
						var_nums=self.data_processor.var_nums)

				val_ac, equ_ac, ans_ac, _, _, test_ans = compute_equations_result(test_res, test_item.output_tokens, self.data_processor.output_lang,
						test_item.nums, test_item.num_stack, ans_list=target_ans, tree=True, prefix=True)

				if test_ans is not None and len(test_ans) > 0:
					if self.canonical(test_ans[0]) == self.canonical(target_ans[0]):
						correct += 1

				if val_ac:
					print('The values computed from train and test expressions match')
					value_ac += 1
				if ans_ac:
					print('The predicted answers match the correct ones')
					answer_ac += 1
				if equ_ac:
					print('The predicted equations match the target ones')
					equation_ac += 1
				if val_ac or ans_ac or equ_ac:
					print('Predicted:', test_ans, ', target:', target_ans)
				eval_total += 1

			except Exception as e:
				traceback.print_exc()
				eval_total += 1
				skipped += 1

		print('{} equation-derived answer accuracy = {:.3f}'.format(mode, float(value_ac) / eval_total))
		print('{} known answer accuracy = {:.3f}'.format(mode, float(answer_ac) / eval_total))
		print('{} equation match accuracy = {:.3f}'.format(mode, float(equation_ac) / eval_total))
		accuracy = float(correct) / eval_total
		print('{} correct answers = {:.3f}'.format(mode, accuracy))
		print('{} skipped = {}'.format(mode, skipped))
		print('Validation time: {:.2f} minutes'.format((time.time() - start) / 60))
		return accuracy


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--MATH-dataroot', default='../dataset/kaggle-dataset/train/*/*', type=str)
	parser.add_argument('--aug-dataset-path', default='../dataset/mawps_combine.json', type=str)

	parser.add_argument('--learning-rate', type=float, default=1e-3)
	parser.add_argument('--seed', type=int, default=13, help='torch manual random number generator seed')
	parser.add_argument('--init-weight', type=float, default=0.08, help='initailization weight')

	parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
	parser.add_argument('--max-epochs', type=int, default=100, help='number of full passes through the training data')
	parser.add_argument('--min-freq', type=int, default=1, help='minimum frequency for vocabulary')
	parser.add_argument('--grad-clip', type=int, default=5, help='clip gradients at this value')

	parser.add_argument('--batch-size', type=int, default=8, help='the size of one mini-batch')
	parser.add_argument('--hidden-size', type=int, default=768, help='hidden size')

	parser.add_argument('--emb-learning-rate', type=float, default=1e-5, help='Learning rate to train embeddings')
	parser.add_argument('--freeze-emb', type=bool, default=False, help='Freeze embedding weights')
	parser.add_argument('--embedding-size', type=int, default=768, help='embedding size')

	parser.add_argument('--beam-size', type=int, default=5, help='the beam size of beam search')
	parser.add_argument("--beam-search", type=bool, default=True, help="whether to use beam search")

	parser.add_argument('--cross-validation', type=bool, default=True, help='whether to perform cross validation')

	cfg = parser.parse_args()


	print('=================================')
	print('Reading data...')
	raw_samples = read_files(cfg.MATH_dataroot, aug_dataset_path=None) #cfg.aug_dataset_path)

	runner = Trainer(cfg, raw_samples)

	if cfg.cross_validation:
		print('Creating folds...')
		folds = prepare_folds(runner.data_processor.pairs, cfg.seed)
	else:
		# Represent as fold in order to use identical code below.
		folds = [(runner.data_processor.pairs, [])]

	for fold_index, fold in enumerate(folds):
		print('Starting fold {}'.format(fold_index + 1))
		if is_gpu_available():
			torch.cuda.empty_cache()
		start = time.time()

		train_pairs, test_pairs = fold
		runner.initialize(train_pairs, test_pairs)
		best_acc = runner.train()

		if len(test_pairs) > 0:
			print('Best validation accuracy for fold {} is {:.3f}'.format(fold_index + 1, best_acc[1]))

		end = time.time()
		print('total time for fold {}: {} minutes\n'.format(fold_index + 1, (end - start) / 60))


if __name__ == '__main__':
	main()
