"""Helper class for working with data.

Some functions are borrowed with modifications from: https://github.com/QinJinghui/SAU-Solver
"""

import random
import json
import copy
import re

import torch

from sympy.parsing.latex import parse_latex


PAD_token = 0

def text2int(text, numwords={}):
    if not numwords:
        units = [
          "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
          "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        for idx, word in enumerate(units):    numwords[word] = (1, idx)
        for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    new_words = []
    start = 0
    words = re.split(r'(\W+)', text)

    for index, word in enumerate(words):
        if word not in numwords:
            if result + current > 0:
                new_words.append(str(result + current))
                current = result = 0
            elif result + current == 0 and index > 0 and words[index - 1] == 'zero':
                new_words.append(str(0))

            new_words.append(word)
            continue

        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    if result + current > 0:
        new_words.append(str(result + current))

    return ''.join(new_words)

def remove_latex_text(text):
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

def clean_text(text, replace_text_nums=False, remove_text=True):
    """Removes unnecessary symbols."""
    text = text.replace('\n', ' ')
    text = text.replace('\\qquad', ' ')
    text = text.replace('\\dfrac', '\\frac')
    text = text.replace('\\tfrac', '\\frac')
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
    text = text.replace('\\$', ' dollar ')
    text = text.replace('\\Rightarrow', ';')
    text = text.replace('**', '^')

    if replace_text_nums:
        new_text = ''
        formula_pattern = re.compile('\\\\\[.+\\\\\]')
        match = formula_pattern.search(text)
        start = 0
        while match is not None:
            new_text += text2int(text[start:match.start()].lower())
            new_text += ' ' + match.group(0) + ' '
            start = match.end()
            match = formula_pattern.search(text, match.end())
        new_text += text2int(text[start:].lower())
        text = new_text
    if remove_text:
        # Remove \\text{...}
        text = remove_latex_text(text)

    return text

def get_raw_equations(text, not_dollars=False):
    """Collects equations from the given text."""
    raw_equations = text.split('$')[1::2]
    if len(raw_equations) == 0 or not_dollars:
        pattern = re.compile('\\\\\[(.+?)\\\\\]')
        raw_equations = pattern.findall(text)

    return raw_equations

def remove_boxed(text):
    """Removes \\boxed from the given text."""
    boxed_pattern = re.compile('\\\\boxed{-?\d+}')
    number_pattern = re.compile('-?\d+')

    boxed_match = boxed_pattern.search(text)
    if boxed_match is not None:
        number_match = number_pattern.search(boxed_match.group(0))
        if number_match is not None:
            eq = number_match.group(0)
            text = text[:boxed_match.start()] + eq + text[boxed_match.end():]

    return text

def replace_fracs(text):
    """Replaces latex fractions with fractions with division sign in the given text."""
    text = self.replace_frac(text)
    num_let_pattern_str = '(\d{1}|[a-zA-Z]{1})'
    text = self.replace_frac(
        text,
        frac_pattern=re.compile('\\\\frac' + num_let_pattern_str + num_let_pattern_str),
        number_in_frac_pattern=re.compile(num_let_pattern_str)
    )
    return text

number_in_frac_pattern_str = '{(\d+|[a-zA-Z]+)}' # in case when variables in fraction
frac_pattern_str = '\\\\frac' + number_in_frac_pattern_str + number_in_frac_pattern_str
frac_pattern = re.compile(frac_pattern_str)
number_in_frac_pattern = re.compile(number_in_frac_pattern_str)    

def replace_frac(text, frac_pattern=frac_pattern, number_in_frac_pattern=number_in_frac_pattern):
    """Replaces fractions using the given fraction pattern."""
    changed_text = ''

    frac_match = frac_pattern.search(text)
    start = 0
    while frac_match is not None:
        number_in_frac_match = number_in_frac_pattern.findall(frac_match.group(0).replace('\\frac', ''))
        if len(number_in_frac_match) == 2:
            changed_text += text[start:frac_match.start()] + '(' + number_in_frac_match[0] + \
                '/' + number_in_frac_match[1] + ')'
            start = frac_match.end()
        else:
            print('Invalid fraction!', text)
            break
        frac_match = frac_pattern.search(text, frac_match.end()) 
    changed_text += text[start:] 

    return changed_text

def rewrite_proportions(equation):
    """Works with equations of form a / b = c / d"""
    # Assuming only one variable in equation.
    frac_match = frac_pattern.findall(equation)

    if len(frac_match) == 2:
        params = [None] * 4
        params[0], params[1] = frac_match[0]
        params[2], params[3] = frac_match[1]
        not_digits_index = -1
        for index, param in enumerate(params):
            if not param.isdigit():
                not_digits_index = index
                break
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

def add_mult_operation(equation):
    """Adds explicit multiplication operation."""
    patterns = [re.compile('\)\('), re.compile('[a-zA-Z]{2}')]
    for pattern in patterns:
        match = pattern.search(equation)
        changed_equation = ''
        start = 0
        while match is not None:
            changed_equation += equation[start:match.start() + 1] + '*' + equation[match.start() + 1: match.end()]
            start = match.end()
            match = pattern.search(equation, match.end())
        changed_equation += equation[start:]
        equation = changed_equation

    return changed_equation

def extract_equations(equations, solution, preprocess):
    """
    Checks the given equations if they have equality sign.
    Performs some cleaning in the equations.
    Returns only those equations which are considered valid.
    """
    single_equations = []
    for equation in equations[::-1]:
        if '=' not in equation:
            continue

        splitted = equation.split('=')
        if len(splitted) > 2:
            for i in range(len(splitted) - 1, 0, -1):
                single_equations.append(splitted[i - 1] + '=' + splitted[i])
        else:
            single_equations.append(equation)

    clean_equations = []

    for equation in single_equations:
        equation = remove_boxed(equation)

        if preprocess:
            parsed = parse_equation(equation)
        else:
            parsed = None
        if parsed is not None:
            clean_equations.append(parsed)
        else:
            clean_equations.append(equation)

    return clean_equations

def find_equation(element, equations):
    """Tries to find the equation for the given element."""
    for index, equation in enumerate(equations):
        left, right = equation.split('=')
        if left == element:
            return index, right
        elif right == element:
            return index, left
    return -1, None

def replacer(element, equations, nums_from_statement):
    # Gets equation with unknown vars or numbers, tries to find those in other equations.
    # Returns equation without vars or unknown numbers if possible.

    equations_to_find = equations
    continue_external_loop = False
    while True:
        if len(equations_to_find) == 0:
            raise Exception('Can\'t build expression', element)
        eq_index, eq = self.find_equation(element, equations_to_find)
        if eq is None:
            raise Exception('Bad, couldn\'t find equation.', element)

        num_var_pattern = re.compile('\d+|[a-zA-Z]+')
        expression = ''
        match = num_var_pattern.search(eq)
        start = 0
        while match is not None:
            expression += eq[start:match.start()]
            replacement = None
            if match.group(0) in nums_from_statement:
                replacement = eq[match.start():match.end()]
            else:
                try:
                    replacement = self.replacer(match.group(0), equations_to_find[eq_index + 1:], nums_from_statement)
                except Exception as e:
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
    """Tries to build the solution equation."""
    exp = 'X = ' + self.replacer(equations[0].split('=')[1], equations[1:], nums_from_statement)

    return exp

def remove_brackets(x):
    """Removes the superfluous brackets."""
    y = x
    if x[0] == "(" and x[-1] == ")":
        x = x[1:-1]
        flag = True
        count = 0
        for s in x:
            if s == ")":
                count -= 1
                if count < 0:
                    flag = False
                    break
            elif s == "(":
                count += 1
        if flag:
            return x
    return y

def indexes_from_sentence(lang, sentence, tree=False):
    """Returns a list of indexes, one for each word in the sentence, plus EOS."""
    res = []
    for word in sentence:
        if len(word) == 0:
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
        else:
            res.append(lang.word2index["UNK"])
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res

def sentence_from_indexes(lang, indices):
    """Returns a list of words from the given indices."""
    sent = []
    for ind in indices:
        sent.append(lang.index2word[ind])
    return sent

def pad_seq(seq, seq_len, max_length):
    """Pad a with the PAD symbol."""
    seq += [PAD_token for _ in range(max_length - seq_len)]
    return seq

def stack_to_string(stack):
    op = ""
    for i in stack:
        if op == "":
            op = op + i
        else:
            op = op + ' ' + i
    return op

def index_batch_to_words(input_batch, input_length, lang):
    """Converts input_batch with token ids into word tokens."""

    contextual_input = []
    for i in range(len(input_batch)):
        contextual_input.append(stack_to_string(sentence_from_indexes(lang, input_batch[i][:input_length[i]])))

    return contextual_input



def find_equations_in_statement(sample):
    """Finds if problem statement contains equations that needs to be solved."""
    problem = sample.problem
    smaller_problem = problem.replace('solution', '')
    smaller_problem = problem.replace('equation', '')
    keywords = ['solve', 'evaluate']
    for keyword in keywords:
        if keyword in problem:
            smaller_problem = problem.replace(keyword, '')
            not_equations_len = sum([len(x) for x in smaller_problem.split('$')[0::2]])

            if not_equations_len > 60:
                continue
            equations = get_final_equations(sample)
            if len(equations) == 0:
                return problem, []
            return keyword + ' ' + ' ; '.join(equations), equations
    return problem, []

def parse_equation(equation):
    """Parses the given equation using SymPy latex parser."""
    try:
        parsed = parse_latex(equation)
        parsed = str(parsed)[3:-1]
        lhs, rhs = parsed.split(',')
        eq = lhs + ' = ' + rhs
        for op in '*+-/()^':
            eq = eq.replace(op, ' ' + op + ' ')
        return eq
    except Exception as e:
        print(e)
        return None   

def get_final_equations(sample):
    raw_equations = sample.get_raw_equations(sample.problem)
    equations = sample.extract_equations(raw_equations, sample.problem)
        
    problem = sample.problem
    for i in range(2):
        if len(equations) == 0:
            raw_equations = sample.get_raw_equations(sample.problem, not_dollars=True)
            equations = sample.extract_equations(raw_equations, sample.problem)
            continue
        break

    final_eqs = []
    for eq in equations:
        parsed = parse_equation(eq)
        if parsed is not None:
            final_eqs.append(parsed)
    return final_eqs    
