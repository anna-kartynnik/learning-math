import regex as re

from .data_utils import remove_brackets


number_in_frac_pattern_str = '{(\d+|[a-zA-Z]+)}' # in case when variables in fraction
frac_pattern_str = '\\\\frac' + number_in_frac_pattern_str + number_in_frac_pattern_str
frac_pattern = re.compile(frac_pattern_str)
number_in_frac_pattern = re.compile(number_in_frac_pattern_str)

class MATHProblem(object):
	OP_PATTERN = re.compile('\\\\blah') #\\\\sqrt|\\\\log')

	def __init__(self, filename, problem, category, level, explanation=None, solution=None, equation=None, json_data=None):
		#print('equation ', equation)

		self.filename = filename
		self.problem = self.clean_text(problem, remove_dollars=True, replace_text_nums=True)
		self.category = category.strip()
		self.level = level.strip()
		self.explanation = self.clean_text(explanation, replace_text_nums=False)
		self.solution = [solution.strip()]
		self.equation = equation.strip()
		self.no_expression = True

		if json_data is not None and 'equation2' in json_data:
			self.equation = json_data['equation2']
			if 'solution2' in json_data:
				self.solution = json_data['solution2']
			self.no_expression = False



		#print(self.equation)

		
		#print(self.var_nums)

		if self.no_expression:

			raw_equations = self.get_raw_equations(self.explanation)
			equations = self.extract_equations(raw_equations, self.solution)

			try:
				self.equation = self.build_expression(equations, self.get_numbers(self.problem))
				#self.var_nums = ['X']
				self.no_expression = False
			except Exception as e:
				#print('error ', e)
				self.no_expression = True

		
		seg = self.problem.strip().split()
		new_seg = []
		for s in seg:
			if len(s) == 1 and s in ",.?!;":
				continue
			new_seg.append(s)
		self.problem = ' '.join(new_seg)

		if self.no_expression:
			#print(filename)
			x =  ' ; '.join(equations) # self.equation

			#print(x)
			self.var_nums = list(set(re.compile('[a-zA-Z]+').findall(x)))
			var_nums_str = ''.join(self.var_nums)
			#print(self.var_nums)

			if len(set(x) - set("0123456789.+-*/^()=; " + var_nums_str + var_nums_str.upper())) != 0:
				#print('first check')
				#print(set("0123456789.+-*/^\\sqrt\\log{}()=; " + var_nums_str + var_nums_str.upper()))
				#print(set(x) - set("0123456789.+-*/^\\sqrt\\log{}()=; " + var_nums_str + var_nums_str.upper()))
				self.no_expression = True
			else:
				eqs = x.split(';')
				#print('eqs ', eqs)
				new_eqs = []
				for eq in eqs:
					sub_eqs = eq.strip().split('=')
					if len(sub_eqs) != 2 or not sub_eqs[0].strip() or not sub_eqs[1].strip():
						#print('not 2 ', eq)
						continue
					new_sub_eqs = []
					for s_eq in sub_eqs:
						try:
							new_sub_eqs.append(remove_brackets(s_eq.strip()))
						except IndexError as e:
							#print(e, equation)
							#print('not 3 ', eq)
							pass
					new_eqs.append(new_sub_eqs[0] + ' = ' + new_sub_eqs[1])
				#print('new_eqs ', new_eqs)
				
				if len(new_eqs) == 1:
					self.equation = new_eqs[0]
				else:
					self.equation = ' ; '.join(new_eqs)
				self.no_expression = False

		temp_eqs = []
		for eq in self.equation.split(';'):
			temp_eqs.append(eq.strip().lower())
		if len(temp_eqs) == 1:
			self.equation = temp_eqs[0]
		else:
			self.equation = ' ; '.join(temp_eqs)

		# Update var_nums.
		op_match = MATHProblem.OP_PATTERN.search(self.equation)
		var_nums = []

		if op_match is None:
			var_nums += re.compile('[a-zA-Z]+').findall(self.equation)
		else:
			start = 0
			while op_match is not None:
				var_nums += re.compile('[a-zA-Z]+').findall(self.equation[start : op_match.start()])
				start = op_match.end()
				op_match = MATHProblem.OP_PATTERN.search(self.equation, op_match.end())

			var_nums += re.compile('[a-zA-Z]+').findall(self.equation[start:])


		self.var_nums = list(set(var_nums))

		if len(self.var_nums) == 0:
			self.equation = 'x = ' + self.solution[0] + ' ; ' + self.equation
			self.var_nums = ['x']

		#print('var_nums ', self.var_nums)
		#print('final eq ', self.equation)

	def get_raw_equations(self, text):
		raw_equations = text.split('$')[1::2]
		if len(raw_equations) == 0:
			pattern = re.compile('\\\\\[(.*)\\\\\]')
			raw_equations = pattern.findall(text)

		return raw_equations


	def text2int(self, textnum, numwords={}):
		if not numwords:
		  units = [
			"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
			"nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
			"sixteen", "seventeen", "eighteen", "nineteen",
		  ]

		  tens = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

		  scales = ["hundred", "thousand", "million", "billion", "trillion"]

		  #numwords["and"] = (1, 0)
		  for idx, word in enumerate(units):    numwords[word] = (1, idx)
		  for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
		  for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

		current = result = 0
		new_words = []
		start = 0
		words = re.split(r'(\W+)', textnum) #word_tokenize(textnum)
		#print(numwords)
		#print(words)
		for index, word in enumerate(words):
			if word not in numwords:
				if result + current > 0:
					new_words.append(str(result + current))
					current = result = 0
				elif result + current == 0 and index > 0 and words[index - 1] == 'zero':
					new_words.append(str(0))

				new_words.append(word)
				continue
			#print(word)
			#print(numwords[word])
			scale, increment = numwords[word]
			current = current * scale + increment
			if scale > 100:
				result += current
				current = 0

		if result + current > 0:
			new_words.append(str(result + current))

		#print(new_words)
		return ''.join(new_words)

	def remove_latex_text(self, text):
		"""Removes unnecessary symbols."""
		pattern = re.compile('\\\\text(rm|bf|it|normal)?\{.*?\}')
		match = pattern.search(text)
		start = 0
		new_text = ''
		while match is not None:
			curly_index = match.group(0).index('{')
			new_text += text[start:match.start()] + text[match.start() + curly_index + 1: match.end() - 1].strip()
			start = match.end()
			match = pattern.search(text, match.end())
		new_text += text[start:]

		return new_text

	def clean_text(self, text, remove_dollars=False, replace_text_nums=False, remove_text=True):
		text = text.replace('\n', ' ')
		text = text.replace('\\qquad', ' ')
		text = text.replace('\\dfrac', '\\frac')
		text = text.replace('\\left(', '(')
		text = text.replace('\\right)', ')')
		text = text.replace('\\left[', '[')
		text = text.replace('\\right]', ']')
		text = text.replace('\\left{', '{')
		text = text.replace('\\right}', '}')
		text = text.replace('\\left\\(', '(')
		text = text.replace('\\right\\)', ')')
		text = text.replace('\\left\\[', '[')
		text = text.replace('\\right\\]', ']')
		text = text.replace('\\left\\{', '{')
		text = text.replace('\\right\\}', '}')
		text = text.replace('\\dots', '...')
		text = text.replace('\\cdots', '...')
		text = text.replace('\\ldots', '...')
		text = text.replace('\\cdot', '*')
		text = text.replace('\\times', '*')	
		text = text.replace('\\!', '')
		#text = text.replace('\\%', ' percent ')
		text = text.replace('\\$', ' dollar ') # [TODO] replace by smth and then return or add in words?
		text = text.replace('\\Rightarrow', ';')
		text = text.replace('**', '^')

		# if remove_dollars:
		# 	text = text.replace('$', '') # Remove formulas from statement?
		if replace_text_nums:
			new_text = ''
			formula_pattern = re.compile('\\\\\[.+\\\\\]')
			match = formula_pattern.search(text)
			start = 0
			while match is not None:
				new_text += self.text2int(text[start:match.start()].lower())
				new_text += ' ' + match.group(0) + ' '
				start = match.end()
				match = formula_pattern.search(text, match.end())
			new_text += self.text2int(text[start:].lower())
			text = new_text
		if remove_text:
			# Remove \\text{...}
			text = self.remove_latex_text(text)

		text = self.replace_fracs(text)

		return text

	def remove_boxed(self, text):
		boxed_pattern = re.compile('\\\\boxed{-?\d+}')
		number_pattern = re.compile('-?\d+')

		boxed_match = boxed_pattern.search(text)
		if boxed_match is not None:
			number_match = number_pattern.search(boxed_match.group(0))
			if number_match is not None:
				eq = number_match.group(0)
				# if '=' not in text:
				# 	eq = 'X=' + eq
				text = text[:boxed_match.start()] + eq + text[boxed_match.end():]
			# print(boxed_match)
			# print(equation[:boxed_match.start()])
			# print(equation[boxed_match.end():])
			# print(re.search(re.compile('\d+'), boxed_match.group(0)))
		return text

	def replace_fracs(self, text):
		text = self.replace_frac(text)
		num_let_pattern_str = '(\d{1}|[a-zA-Z]{1})'
		text = self.replace_frac(
			text,
			frac_pattern=re.compile('\\\\frac' + num_let_pattern_str + num_let_pattern_str),
			number_in_frac_pattern=re.compile(num_let_pattern_str)
		)
		return text		

	def replace_frac(self, text, frac_pattern=frac_pattern, number_in_frac_pattern=number_in_frac_pattern):
		
		changed_text = ''

		frac_match = frac_pattern.search(text)
		start = 0
		while frac_match is not None:
			#print(frac_match)
			number_in_frac_match = number_in_frac_pattern.findall(frac_match.group(0).replace('\\frac', ''))
			#print(number_in_frac_match)
			if len(number_in_frac_match) == 2:
				changed_text += text[start:frac_match.start()] + '(' + number_in_frac_match[0] + \
					'/' + number_in_frac_match[1] + ')'
				start = frac_match.end()
			else:
				print('Invalid fraction!', text) # or raise?
				break
			frac_match = frac_pattern.search(text, frac_match.end()) 
		changed_text += text[start:] 

		return changed_text

	def rewrite_proportions(self, equation):
		"""Works with equations of form a / b = c / d"""
		# Assuming only one variable in equation.
		frac_match = frac_pattern.findall(equation)
		#print(frac_match)
		if len(frac_match) == 2:
			params = [None] * 4
			params[0], params[1] = frac_match[0]
			params[2], params[3] = frac_match[1]
			#print(params)
			not_digits_index = -1
			for index, param in enumerate(params):
				if not param.isdigit():
					not_digits_index = index
					break # or not break but we assume one variable max
			if not_digits_index == -1:
				# No variables, nothing to do.
				return equation
			else:
				if not_digits_index == 0:
					frac_ind = (2, 3)
					mult_ind = 1
				elif not_digits_index == 2:
					frac_ind = (0, 1)
					mult_ind = 3
				elif not_digits_index == 1:
					frac_ind = (3, 2)
					mult_ind = 0
				else:
					frac_ind = (1, 0)
					mult_ind = 2
				frac = '\\frac{' + params[frac_ind[0]] + '}{' + params[frac_ind[1]] + '}'
				equation = params[not_digits_index] + '=(' + frac + ')*(' + params[mult_ind] + ')'


		return equation

	def _add_mult_operation(self, equation):
		patterns = [re.compile('\)\('), re.compile('[a-zA-Z]{2}')]
		for pattern in patterns:
			match = pattern.search(equation)
			changed_equation = ''
			start = 0
			while match is not None:
				#print(match)
				changed_equation += equation[start:match.start() + 1] + '*' + equation[match.start() + 1: match.end()]
				start = match.end()
				match = pattern.search(equation, match.end())
			changed_equation += equation[start:]
			equation = changed_equation

		return changed_equation 

	def add_mult_operation(self, equation):
		# TODO exclude operations like log or cos...
		op_match = MATHProblem.OP_PATTERN.search(equation)
		new_equation = ''

		if op_match is None:
			return self._add_mult_operation(equation)
		else:
			start = 0
			while op_match is not None:
				new_equation += self._add_mult_operation(equation[start:op_match.start()]) + equation[op_match.start():op_match.end()]
				start = op_match.end()
				op_match = MATHProblem.OP_PATTERN.search(equation, op_match.end())

			new_equation += self._add_mult_operation(equation[start:])

		return new_equation


	def extract_equations(self, equations, solution):
		single_equations = []
		#print(equations)
		for equation in equations[::-1]:
			if '=' not in equation: # and '\\boxed' not in equation:
				continue

			splitted = equation.split('=')
			if len(splitted) > 2:
				for i in range(len(splitted) - 1, 0, -1):
					single_equations.append(splitted[i - 1] + '=' + splitted[i])
			else:
				single_equations.append(equation)
		#print(single_equations)

		clean_equations = []
		# if len(single_equations) == 0 or not single_equations[0].startswith('X='):
		# 	clean_equations.append('X=' + solution)
		for equation in single_equations:
			equation = self.remove_boxed(equation)
			#print(equation)
			#equation = self.rewrite_proportions(equation)
			equation = self.replace_frac(equation)
			equation = self.add_mult_operation(equation)
			equation = equation.replace('[', '(')
			equation = equation.replace(']', ')')

			for op in '=+-/*^()':
				equation = equation.replace(op, ' ' + op + ' ')

			clean_equations.append(equation)

		#print(clean_equations)

		return clean_equations

	def find_equation(self, element, equations):
		for index, equation in enumerate(equations):
			left, right = equation.split('=')
			if left == element:
				return index, right
			elif right == element:
				return index, left
		return -1, None

	def replacer(self, element, equations, nums_from_statement):
		# Gets equation with unknown vars or numbers, tries to find those in other equations.
		# Returns equation without vars or unknown numbers if possible.
		#print('Finding replacer for ', element)
		equations_to_find = equations
		continue_external_loop = False
		while True:
			if len(equations_to_find) == 0:
				raise Exception('Can\'t build expression', element)
			eq_index, eq = self.find_equation(element, equations_to_find)
			if eq is None:
				raise Exception('Bad, couldn\'t find equation.', element) # raise?

			num_var_pattern = re.compile('\d+|[a-zA-Z]+')
			expression = ''
			match = num_var_pattern.search(eq)
			start = 0
			while match is not None:
				#print(match)
				expression += eq[start:match.start()]
				replacement = None
				if match.group(0) in nums_from_statement:
					replacement = eq[match.start():match.end()]
				else:
					try:
						replacement = self.replacer(match.group(0), equations_to_find[eq_index + 1:], nums_from_statement)
					except Exception as e:
						#print(e)
						equations_to_find = equations_to_find[eq_index + 1:]
						continue_external_loop = True
						break

				expression += replacement
				start = match.end()
				match = num_var_pattern.search(eq, match.end())

			if continue_external_loop:
				continue_external_loop = False
				continue
			expression += eq[start:]
			return expression  

	def build_expression(self, equations, nums_from_statement):
		#print('building exp')
		exp = 'X = ' + self.replacer(equations[0].split('=')[1], equations[1:], nums_from_statement)
		#print('eexp ', exp)
		return exp

	def get_numbers(self, problem):
		number_pattern = re.compile('\d+')
		numbers = number_pattern.findall(problem)
		if 'percent' in problem or '%' in problem:
			numbers.append(100)

		return numbers
