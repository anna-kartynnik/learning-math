import torch


def is_gpu_available():
	"""Returns `True` if GPU is available."""
	return torch.cuda.is_available()

def get_available_device():
	"""Returns available device."""
	if is_gpu_available():
		device = torch.device('cuda')

		print('There are %d GPU(s) available.' % torch.cuda.device_count())
		print('We will use the GPU:', torch.cuda.get_device_name(0))

	else:
		print('No GPU available, using the CPU instead.')
		device = torch.device('cpu')

	return device

