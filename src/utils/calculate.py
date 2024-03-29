"""Builds and evaluates expression trees; solves equation systems via SymPy.

Borrowed with modifications from: https://github.com/QinJinghui/SAU-Solver
"""


import re
from copy import deepcopy
from sympy import S, Eq, solve, solve_poly_system
from sympy.parsing.sympy_parser import parse_expr
import copy
from expression_tree import *
from wrapt_timeout_decorator import timeout


_TIMEOUT_SEC = 5


@timeout(_TIMEOUT_SEC, use_signals=True)
def solve_linear_equation_with_one_unknown(equ_str, var_str):
    x = S(var_str.split())
    equ_str_split = equ_str.split('=')
    if len(equ_str_split) == 1:
        equ_str = equ_str_split[0]
        equ_expr = parse_expr(equ_str)
    else:
        equ_str= equ_str_split[0] + ' - ( ' + equ_str_split[1] + ' )'
        equ_expr = parse_expr(equ_str)
    ans = solve(equ_expr, x)
    return ans[0]


@timeout(_TIMEOUT_SEC, use_signals=True)
def solve_linear_equation_with_multiple_unknown(equ_str_list, var_str):
    var_list = S(var_str.split())
    equ_list = []
    for equ_str in equ_str_list:
        equ_str_split = equ_str.split('=')
        if len(equ_str_split) == 1:
            equ_list.append(parse_expr(equ_str_split[0]))
        else:
            equ_list.append(equ_str_split[0] + ' - ( ' + equ_str_split[1] + ' )')
    ans = solve(equ_list, var_list)

    if isinstance(ans, dict):
        return list(ans.values())
    elif isinstance(ans, list):
        new_ans = []
        for a in ans:
            if isinstance(a, tuple):
                new_ans.extend(a)
            else:
                new_ans.append(a)
        return new_ans


def compute_infix_expression(infix_expression):
    single_flag = True
    if 'x' in infix_expression and 'y' in infix_expression:
        single_flag = False
        expression_str = ' '.join(infix_expression)
        try:
            expression_list = expression_str.split(';')
            answers = solve_linear_equation_with_multiple_unknown(expression_list, 'x y')
            if len(answers) == 1:
                answers = "error"
        except:
            answers = "error"
    else:
        expression_str = ' '.join(infix_expression)
        if 'x' in expression_str:
            var_str = 'x'
        else:
            var_str = 'y'
        try:
            expression_list = expression_str.split(';')
            answers = solve_linear_equation_with_multiple_unknown(expression_list[0], var_str)
        except:
            answers = "error"

    return answers, single_flag


def compute_postfix_expression(postfix_expression):
    st = list() # stack
    operators = ["+", "-", "^", "*", "/"]
    for p in postfix_expression:
        if p not in operators:
            pos = re.search("\d+\(", p)
            if pos:
                st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
            elif p[-1] == "%":
                st.append(float(p[:-1]) / 100)
            else:
                st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if a == 0:
                return None
            st.append(b / a)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(b - a)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None


def compute_prefix_expression(pre_fix):
    st = list()  # stack
    operators = ["+", "-", "^", "*", "/"]
    pre_fix = deepcopy(pre_fix)
    pre_fix.reverse()
    for p in pre_fix:
        if p not in operators:
            pos = re.search("\d+\(", p)
            if pos:
                st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
            elif p[-1] == "%":
                st.append(float(p[:-1]) / 100)
            else:
                st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
                return None
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None


def out_expression_list(test, output_lang, num_list, num_stack=None):
    max_index = output_lang.n_words
    res = []
    for i in test:
        if i < max_index - 1:
            idx = output_lang.index2word[i]
            if idx[0] == "N":
                if int(idx[1:]) >= len(num_list):
                    return None
                if "%" in num_list[int(idx[1:])]:
                    res.append('(' + num_list[int(idx[1:])][:-1] + '/100)')
                else:
                    res.append(num_list[int(idx[1:])])
            else:
                res.append(idx)
        else:
            if num_stack is None or len(num_stack) == 0:
                return ""
            pos_list = num_stack.pop()
            if len(pos_list) == 0 or len(num_list) < pos_list[0]:
                print('Empty pos_list or not enough nums')
                print(pos_list)
                print(num_list)
                print(test)
                return ""
            c = num_list[pos_list[0]]
            res.append(c)
    return res


def out_equation(test, output_lang, num_list, num_stack=None):
    test = test[:-1]
    max_index = len(output_lang.index2word) - 1
    test_str = ""
    for i in test:
        if i < max_index:
            c = output_lang.index2word[i]
            if c == "^":
                test_str += "**"
            elif c == "[":
                test_str += "("
            elif c == "]":
                test_str += ")"
            elif c[0] == "N":
                if int(c[1:]) >= len(num_list):
                    return None
                x = num_list[int(c[1:])]
                if x[-1] == "%":
                    test_str += "(" + x[:-1] + "/100" + ")"
                else:
                    test_str += x
            else:
                test_str += c
        else:
            if len(num_stack) == 0:
                print(test_str, num_list)
                return ""
            n_pos = num_stack.pop()
            test_str += num_list[n_pos[0]]
    return test_str


def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_postfix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_postfix_expression(test) - compute_postfix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_result(test_res, test_tar, output_lang, num_list, num_stack):
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True
    test = out_equation(test_res, output_lang, num_list)
    tar = out_equation(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False
    if test == tar:
        return True, True
    try:
        if abs(eval(test) - eval(tar)) < 1e-4:
            return True, False
        else:
            return False, False
    except:
        return False, False


def close(value_a, value_b, tolerance=1e-1):
    try:
        return abs(float(value_a) - float(value_b)) < tolerance
    except ValueError:
        # Couldn't convert to `float`.
        return False


def close_to_subset(values_a, values_b, tolerance=1e-1):
    return all(any(close(value_a, value_b) for value_b in values_b)
               for value_a in values_a)


def compute_equations_result(test_res, test_tar, output_lang, num_list, num_stack,
                             ans_list=None, tree=False, prefix=False, infix=False, postfix=False):
    if ans_list is None:
        ans_list = []
    if not isinstance(test_res, list):
        test_res = test_res.tolist()
    if len(num_stack) == 0 and test_res == test_tar:
        test = out_expression_list(test_res, output_lang, num_list)
        tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
        if not tree or (tree and infix):
            test_ept = ExpressionTree()
            test_ept.build_tree_from_infix_expression(test)
            tar_ept = ExpressionTree()
            tar_ept.build_tree_from_infix_expression(tar)
        elif postfix:
            test_ept = ExpressionTree()
            test_ept.build_tree_from_postfix_expression(test)
            tar_ept = ExpressionTree()
            tar_ept.build_tree_from_postfix_expression(tar)
        else:
            test_ept = ExpressionTree()
            test_ept.build_tree_from_prefix_expression(test)
            tar_ept = ExpressionTree()
            tar_ept.build_tree_from_prefix_expression(tar)
        if not tree or (tree and infix):
            test_expression = test
            tar_expression = tar
        else:
            tar_expression = tar_ept.get_infix_expression()
            test_expression = test_ept.get_infix_expression()
        return True, True, True, test_expression, tar_expression, None
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))

    if test is None:
        if not tree or (tree and infix):
            tar_ept = ExpressionTree()
            tar_ept.build_tree_from_infix_expression(tar)
        elif postfix:
            tar_ept = ExpressionTree()
            tar_ept.build_tree_from_postfix_expression(tar)
        else:
            tar_ept = ExpressionTree()
            tar_ept.build_tree_from_prefix_expression(tar)

        if not tree or (tree and infix):
            tar_expression = tar
        else:
            tar_expression = tar_ept.get_infix_expression()
        return False, False, False, test, tar_expression, None
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    test_var_list = set()
    for elem in test:
        if elem in alphabet:
            test_var_list.add(elem)
    test_var_list = list(test_var_list)
    tar_var_list = set()
    for elem in tar:
        if elem in alphabet:
            tar_var_list.add(elem)
    tar_var_list = list(tar_var_list)
    if not tree or (tree and infix):
        test_ept = ExpressionTree()
        test_ept.build_tree_from_infix_expression(test)
        tar_ept = ExpressionTree()
        tar_ept.build_tree_from_infix_expression(tar)
    elif postfix:
        test_ept = ExpressionTree()
        test_ept.build_tree_from_postfix_expression(test)
        tar_ept = ExpressionTree()
        tar_ept.build_tree_from_postfix_expression(tar)
    else:
        test_ept = ExpressionTree()
        test_ept.build_tree_from_prefix_expression(test)
        tar_ept = ExpressionTree()
        tar_ept.build_tree_from_prefix_expression(tar)

    if not tree or (tree and infix):
        test_expression = test
        tar_expression = tar
    else:
        tar_expression = tar_ept.get_infix_expression()
        test_expression = test_ept.get_infix_expression()
    new_tar_expression = []
    for elem in tar_expression:
        if elem == "^":
            new_tar_expression.append("**")
        else:
            new_tar_expression.append(elem)

    new_test_expression =[]
    for elem in test_expression:
        if elem == "^":
            new_test_expression.append("**")
        else:
            new_test_expression.append(elem)
    tar_expression = new_tar_expression
    test_expression = new_test_expression
    idx = 0
    while idx < len(test_expression):
        if test_expression[idx] == '**' and idx + 1 < len(test_expression):
            try:
                p = float(test_expression[idx+1])
                ip = int(p)
                if p > 3.0 or p - ip != 0:
                    return False, False, False, test_expression, tar_expression, None
            except:
                return False, False, False, test_expression, tar_expression, None
        idx = idx + 1

    test_ans = compute_expressions(test_expression, test_var_list)
    tar_ans = compute_expressions(tar_expression, tar_var_list)

    if tar_ans == 'error':
        tar_ans = ans_list

    if test_ans == 'error' or test_ans == [] or test_ans is None or tar_ans is None or tar_ans == 'error' or tar_ans == []:
        return False, False, False, test_expression, tar_expression, None

    # The computed values for the target and the predicted expressions match.
    val_ac = close_to_subset(test_ans, tar_ans)

    # The equations are accurately predicted.
    equ_ac = test_expression == tar_expression

    # The predicted answers match the known answers.
    ans_ac = True
    if len(ans_list) == 0:
        ans_ac = False
    elif len(ans_list) == 1:
        # Some of the answers among the predicted ones is correct.
        ans_ac = close_to_subset(ans_list, test_ans)
    else:
        # Every predicted answer matches some target answer.
        ans_ac = close_to_subset(test_ans, ans_list)

    return val_ac, equ_ac, ans_ac, test_expression, tar_expression, test_ans


def compute_expressions(test_expression, var_list):
    """Computes expressions with given variables using SymPy."""
    filtered_var_list = set()
    for elem in test_expression:
        if elem in var_list:
            filtered_var_list.add(elem)
    if len(filtered_var_list) == 0:
        test_expression_str = ' '.join(test_expression)
        test_expression_str = 'x = ' + test_expression_str
        filtered_var_list = ['x']
        test_ans = get_result_from_sympy([test_expression_str], filtered_var_list)
    else:
        test_expression_str = ' '.join(test_expression)
        test_expression_strs = test_expression_str.split('SEP')
        test_expression_str_list = []
        for test_expression in test_expression_strs:
            if '=' not in test_expression:
                test_expression_str_list.append("0 = " + test_expression)
            else:
                test_expression_str_list.append(test_expression)
        test_ans = get_result_from_sympy(test_expression_str_list, list(filtered_var_list))
    return test_ans


def get_result_from_sympy(expression_list, var_list):
    expression_str_list = []
    for expression in expression_list:
        expression_str_list.append(expression)
    var_str =' '.join(var_list)
    try:
        gen_ans = solve_linear_equation_with_multiple_unknown(expression_str_list, var_str)
    except Exception as e:
        gen_ans = "error"

    return gen_ans
