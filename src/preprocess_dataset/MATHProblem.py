import regex as re

from .data_utils import clean_text, remove_brackets, find_equations_in_statement, get_raw_equations, extract_equations, build_expression


class MATHProblem(object):

	def __init__(self, filename, problem, category, level, explanation=None, solution=None, equation=None, json_data=None, preprocess=False):
		self.filename = filename
		self.problem = clean_text(problem, replace_text_nums=True)
		self.category = category.strip()
		self.level = level.strip()
		self.explanation = clean_text(explanation, replace_text_nums=False) if explanation is not None else None
		self.solution = [solution.strip()]
		self.equation = equation.strip()
		self.no_expression = True
		self.preprocessed = False

		if json_data is not None and 'equation2' in json_data:
			self.equation = json_data['equation2']
			if 'solution2' in json_data:
				self.solution = json_data['solution2']
			self.no_expression = False

		if preprocess:
			if self.no_expression:
				final_problem, equations = find_equations_in_statement(self)
				if len(equations) > 0:
					self.problem = final_problem
					if len(equations) == 1:
						self.equation = equations[0]
						self.preprocessed = True
					else:
						self.equation = ' ; '.join(equations)
					self.no_expression = False

		equations = self.equation.split(';')

		if self.no_expression and self.explanation is not None:

			raw_equations = get_raw_equations(self.explanation)
			equations = extract_equations(raw_equations, self.solution, preprocess)

			try:
				self.equation = build_expression(equations, self.get_numbers(self.problem))
				self.no_expression = False
			except Exception as e:
				self.no_expression = True

		
		seg = self.problem.strip().split()
		new_seg = []
		for s in seg:
			if len(s) == 1 and s in ",.?!;":
				continue
			new_seg.append(s)
		self.problem = ' '.join(new_seg)

		if self.no_expression:
			x =  ' ; '.join(equations) # self.equation

			self.var_nums = list(set(re.compile('[a-zA-Z]+').findall(x)))
			var_nums_str = ''.join(self.var_nums)

			if len(set(x) - set("0123456789.+-*/^()=; " + var_nums_str + var_nums_str.upper())) != 0:
				self.no_expression = True
			else:
				eqs = x.split(';')
				new_eqs = []
				for eq in eqs:
					sub_eqs = eq.strip().split('=')
					if len(sub_eqs) != 2 or not sub_eqs[0].strip() or not sub_eqs[1].strip():
						continue
					new_sub_eqs = []
					for s_eq in sub_eqs:
						try:
							new_sub_eqs.append(remove_brackets(s_eq.strip()))
						except IndexError as e:
							pass
					new_eqs.append(new_sub_eqs[0] + ' = ' + new_sub_eqs[1])
				
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
		var_nums = re.compile('[a-zA-Z]+').findall(self.equation)

		self.var_nums = list(set(var_nums))

		if len(self.var_nums) == 0:
			self.equation = 'x = ' + self.solution[0] + ' ; ' + self.equation
			self.var_nums = ['x']

