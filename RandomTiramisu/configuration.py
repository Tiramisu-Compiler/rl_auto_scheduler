# from scipy import interpolate
import numpy as np
import random
import re
from sympy import simplify
from sympy.core.symbol import Symbol
from TiramisuMaker import Buffer, ReadAccess
import copy
# from tqdm import tqdm
# import seaborn as sns
# import matplotlib.pyplot as plt

local_random = random.Random()  # create a new random generator to avoid interference with sympy
local_random2 = random.Random()  # this random generator is only used for detecting skippable loops
local_random2.seed(42)

MAX_BUFFER_SIZE = 2**25  # value, which translates as 256MB of float64 values
MAX_ITERATION_SPACE = np.inf  # currently we don't need to set it since the buffer sizes is directly controlled at creation


def choose_datatype():  # choose computation's datatype
    pass


def choose_problem_size():  # chooses the size of the program which is used for defining loop extents and buffer sizes
    # to do choose uniformly between the five sizes of polybench
    # return local_random.choice(['MINI', 'SMALL', 'MEDIUM', 'LARGE', 'XLARGE'])
    return local_random.choice(['SMALL', 'MEDIUM', 'LARGE'])  # Uniform choice
#     return local_random.choice(['MINI'])  # Uniform choice


def choose_nb_root_loops():  # chooses the number of root loops (number of nodes)
    return 1  # currently only programs with one root loop are generated, to be changed later


def choose_write_buf(defining_iterators, program):
    # chooses a buffer where to write the computation
    # can be either a new input buffer or buffer already declared in the program if it matches the parent loops dims
    # if the choice is to create new buf, we need to declare new input?, add buffer to buf list

    candidate_buffers = []
    for buffer in program.buffer_list:
        if len(defining_iterators) == len(buffer.defining_iterators):
            if all(defining_iterators[i] is buffer.defining_iterators[i] for i in range(len(defining_iterators))):
                candidate_buffers.append(buffer)
    candidate_buffers.append('new buffer')
    choice_weights = [1]*(len(candidate_buffers)-1)+[1.2]  # the chance to use a new buffer is 1.2 time the chance to use an existing one
    selected_buffer = local_random.choices(candidate_buffers, choice_weights, k=1)[0]
    if selected_buffer == 'new buffer':
        return Buffer(defining_iterators=defining_iterators, program=program)
    else:
        return selected_buffer


def choose_write_access_pattern(computation):
    # choose a write access pattern for the computation
    # depends on the loop depth
    # at a first step support removing one or two dims from beginning or ending
    # at a second step support transposing accesses / +1 , backward accesses, slice write (with a constant) ...
    # returns an access matrix and a list [j, k, ...] of loops that defines the rows of the access matrix
    comp_iterators = computation.parent_iterators_list
    if len(comp_iterators) >= 2:  # loop depth is at least 2
        # make a list of candidate defining iterators that makes the buffer size not exceed the limit
        # currently the candidates are generated by dropping iterators either from the end or from the beginning of the list
        candidates = []
        if get_iteration_space_from_iterators(comp_iterators) <= MAX_BUFFER_SIZE:
            candidates.append(comp_iterators)
        nb_dropped = 1  # number of dimensions to drop from the beginning
        while get_iteration_space_from_iterators(comp_iterators[nb_dropped:]) > MAX_BUFFER_SIZE:
            nb_dropped += 1
        assert len(comp_iterators[nb_dropped:]) >= 1
        candidates.append(comp_iterators[nb_dropped:])
        nb_dropped = 1  # number of dimensions to drop from the end
        while get_iteration_space_from_iterators(comp_iterators[:-nb_dropped]) > MAX_BUFFER_SIZE:
            nb_dropped += 1
        assert len(comp_iterators[:-nb_dropped]) >= 1
        candidates.append(comp_iterators[:-nb_dropped])

        defining_iterators = local_random.choice(candidates)  # uniformly

    else:
        defining_iterators = comp_iterators
    access_matrix = np.zeros((len(defining_iterators), len(comp_iterators)+1), dtype=int)
    for i in range(access_matrix.shape[0]):
        access_matrix[i, comp_iterators.index(defining_iterators[i])] = 1  # should always diagonal for now
    # np.fill_diagonal(access_matrix, 1)  # always diagonal for now
    return access_matrix, defining_iterators


def get_iteration_space_from_iterators(iterators_list):
    iter_space_size = 1
    for iterator in iterators_list:
        iter_space_size *= (iterator.upper_bound-iterator.lower_bound)
    return iter_space_size


def choose_lower_bound(problem_size):
    return 0  # currently always return zero


def choose_upper_bound(problem_size, parent_upper_bounds):
    # when choosing the extents, can first create a sublist from the plb_sizes[pb_size] to make the chance of choosing square matrices bigger
    MINI_sizes = [16, 32, 48]  # multiples of 16
    SMALL_sizes = [32, 64, 96, 128]  # multiples of 32
    MEDIUM_sizes = [64, 128, 192, 256, 320, 384, 448]  # multiples of 64
    LARGE_sizes = [256, 512, 768, 1024, 1280, 1536, 2048]  # multiples of 256
    XLARGE_sizes = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]  # multiples of 512
    all_sizes = sorted(MINI_sizes+SMALL_sizes+MEDIUM_sizes+LARGE_sizes+XLARGE_sizes)
#     all_sizes = sorted(SMALL_sizes+MEDIUM_sizes+LARGE_sizes)

    bound_flip = local_random.random()
    upper_bound = 0
    if bound_flip < 0.05:  # 0.05 picking from other sizes list
        upper_bound = local_random.choices(all_sizes, weights=[1/i for i in range(1, len(all_sizes)+1)], k=1)[0]
    elif bound_flip < 0.25 and parent_upper_bounds != []:  # 0.2 picking parent's size
        upper_bound = parent_upper_bounds[-1]
    else:  # 0.8 pick from the concerned sizes list
        if problem_size == 'SMALL':
            upper_bound = local_random.choices(SMALL_sizes, weights=[1/i for i in range(1, len(SMALL_sizes)+1)], k=1)[0]
        elif problem_size == 'MEDIUM':
            upper_bound = local_random.choices(MEDIUM_sizes, weights=[1/i for i in range(1, len(MEDIUM_sizes)+1)], k=1)[0]
        elif problem_size == 'LARGE':
            upper_bound = local_random.choices(LARGE_sizes, weights=[1/i for i in range(1, len(LARGE_sizes)+1)], k=1)[0]
        elif problem_size == 'XLARGE':
            upper_bound = local_random.choices(XLARGE_sizes, weights=[1/i for i in range(1, len(XLARGE_sizes)+1)], k=1)[0]
        elif problem_size == 'MINI':
            upper_bound = local_random.choices(MINI_sizes, weights=[1/i for i in range(1, len(MINI_sizes)+1)], k=1)[0]

    # TODO check if the iteration space the concerned loop level doesn't exceed certain threshold
        # TODO if it exceeds pick a smaller size
    # iteration_space_size = MAX_ITERATION_SPACE
    # while iteration_space_size > MAX_ITERATION_SPACE

    iteration_space_size = upper_bound
    for bound in parent_upper_bounds:
        iteration_space_size *= bound

    while iteration_space_size > MAX_ITERATION_SPACE:
        if upper_bound == all_sizes[0]: # smallest size
            # the best is to reduce parent iterators size, this will require to update buffer and input sizes ...
            # temporarily if this happens, nothing is done, we just accept it
            break
        else:
            upper_bound = all_sizes[all_sizes.index(upper_bound)-1]  # take the next smaller size

    return upper_bound


def choose_nb_child_loops(loop):  # chooses the number of child loops to create in a given loop
    # the choice should depend on the loop depth, number of total loops so far in the program, number of loop siblings ?, ..
    # or proba of 0 child depending on the depth and uniform/skew-normal [1,N] independently of the rest

    if loop.depth == 0:  # since currently we can't schedule programs with less than 2 shared loops, we need to force the generation of 2 shared loops
        nb_child_loops = 1
    elif loop.depth == 6:
        nb_child_loops = 0  # since the current representation needs a fixed maximal depth of the loop nest
    # if loop.program.nb_branches == 2:  # no more splits allowed
    #     if loop.depth >= 2:
    #         nb_child_loops = local_random.choices([0, 1], weights=[0.7, 0.3], k=1)[0]  # to stop it from growing exaggeratedly deep
    #     else:
    #         nb_child_loops = local_random.choices([0, 1], weights=[0.4, 0.6], k=1)[0]
    # elif loop.program.nb_branches == 1:
    elif loop.depth >= 2:
        nb_child_loops = round(abs(local_random.gauss(mu=0, sigma=1 / 2.5)))  # to stop it from growing exaggeratedly deep
    else:
        nb_child_loops = round(abs(local_random.gauss(mu=0, sigma=1.1)))
    # else:
    #     print('error nb branches not supported')
    #     return
    return nb_child_loops  # temporarily, to be changed later


def choose_nb_child_comps(parent_loop):  # chooses the number of direct child computations of a given loop
    # should depend on the loop depth, number of loop siblings, + some distrib that limits the number of total comps in the program

    # if parent_loop.depth == 0:  # since computations with one loop can't be scheduled
    #     return 0

    # nb_child_comps = round(abs(local_random.gauss(mu=0, sigma=0.5)))
    #
    # if parent_loop.nb_child_loops == 0 and nb_child_comps == 0:  # leaf loops must have at least one computation
    #     return round(abs(local_random.gauss(mu=0, sigma=0.5))) + 1
    # else:
    #     return nb_child_comps
    if parent_loop.nb_child_loops == 0:  # if leaf loop, generate computations
        return round(abs(local_random.gauss(mu=0, sigma=0.5))) + 1
    else:
        return 0


def choose_child_order(nb_child_loops, nb_child_comps):  # chooses the order of child loops and computation of a given loop
    # returns a list of [('L'|'C')+] where L is loop position and C is computation position
    # should depend on loop depth ?,
    positions_list = ['L']*nb_child_loops + ['C']*nb_child_comps
    local_random.shuffle(positions_list)  # return uniform shuffling
    return positions_list


def choose_read_accesses(computation, program):  # chooses buffers to read and access pattern for each (list of ReadAccess objects)
    # usable existing buffer are buffers where buf.defining_iterators is included in comp.parent_loops
    # use the computation's absolute order to decide whether computation is initialization
    # constraints: all the computation's iterators must be used in its expression EDIT not necessarily e.g. time step iteration
    # TODO is and remove redundant mem accesses, Edit: check if really needed first
    comp_iterators = computation.parent_iterators_list
    candidate_buffers = []
    computation_iterator = set(computation.parent_iterators_list)
    for buffer in program.buffer_list:
        buffer_iterator = set(buffer.defining_iterators)
        if buffer_iterator.issubset(computation_iterator):
            candidate_buffers.append(buffer)

    nb_buffers_to_read = round(random_truncnorm(mu=2.0, sigma=1.1, low=0))
    if computation.absolute_order < 3:  # giving a bit more chance for the 3 first computations to be an initialization
        nb_buffers_to_read = local_random.choices([nb_buffers_to_read, 0], weights=[0.97, 0.03/(computation.absolute_order+1)], k=1)[0]
    candidate_buffers.append('new buffer')

    # weighting the chances of selecting buffers to read
    buffer_selection_weights = []
    for buffer in candidate_buffers:
        if buffer is computation.write_access.buffer:
            buffer_selection_weights.append(1.2)
        elif buffer == 'new buffer':
            buffer_selection_weights.append(1.1)
        else:
            buffer_selection_weights.append(len(buffer.defining_iterators)/len(computation.parent_iterators_list)+0.6)  # give more proba to buffers with more dims

    selected_buffers = local_random.choices(candidate_buffers, weights=buffer_selection_weights, k=nb_buffers_to_read)

    # create the new buffers if needed
    for _ in range(selected_buffers.count('new buffer')):
        # define a new buffer using a permutation of a subset of the computation's iterators
        buffer_size = MAX_BUFFER_SIZE + 1
        while buffer_size > MAX_BUFFER_SIZE:  # re-pick the defining iterators until the buffer size is bellow the limit
            buffer_dim = local_random.choices(population=list(range(1, len(comp_iterators)+1)),
                                        weights=[i/len(comp_iterators) for i in range(1, len(comp_iterators)+1)], k=1)[0]  # give more proba to nb dims closest to computation dim
            selected_iterators = sorted_sample(computation.parent_iterators_list, buffer_dim)
            buffer_size = get_iteration_space_from_iterators(selected_iterators)

        if local_random.random() < 0.15:  # probability of unordered dims for the new buffer
            local_random.shuffle(selected_iterators)
        buffer = Buffer(defining_iterators=selected_iterators, program=program)
        selected_buffers.append(buffer)
        selected_buffers.remove('new buffer')

    # if proba stencil, pick some bufs from the selected and create stencil accesses
    # create random access matrices on the rest of the buffers
    # if local_random.random() < 0.33:  # probability of making a full stencil
    #
    #     pass

    read_accesses_list = []
    for buffer in selected_buffers:  # generate ?random? access matrices for each buffer, the current version is not random, maybe randomness can be generated for stencils
        access_matrix = np.zeros((len(buffer.defining_iterators), len(computation.parent_iterators_list)+1), dtype=int)
        for i in range(len(buffer.defining_iterators)):
            j = computation.parent_iterators_list.index(buffer.defining_iterators[i])
            access_matrix[i, j] = 1

        if local_random.random() < 1/(len(selected_buffers)**2*1.33):  # probability of making a stencil, gives higher chances when number of buffers is small, gives 0.75 for 1, 0.2 for 2
            # TODO check plb's stencil patterns and implement them all and affect probas, use higher proba when nb_bufs is small
            # when making a stencil update the buffers dims
            if len(buffer.defining_iterators) < 3:
                stencil_type = local_random.choices(['star', 'square', 'uni_dim'], weights=[0.4, 0.2, 0.4], k=1)[0]
            else:  # if the dimensions of the buffer are >=3 we avoid generating square stencil since it generate an excessive number of memory reads
                stencil_type = local_random.choices(['star', 'uni_dim'], weights=[0.5, 0.5], k=1)[0]
            if stencil_type == 'star':  # star stencil over all dimensions {3dims(hea3d) or 2dims (jacobi2d)}
                # access_matrices_list = [access_matrix.copy() for _ in range(len(buffer.defining_iterators)+1)]
                access_matrices_list = [access_matrix]
                for i in range(len(buffer.defining_iterators)):
                    access_matrix_copy = access_matrix.copy()
                    access_matrix_copy[i, -1] = 1
                    access_matrices_list.append(access_matrix_copy)
                    access_matrix_copy = access_matrix.copy()
                    access_matrix_copy[i, -1] = -1
                    access_matrices_list.append(access_matrix_copy)
                for matrix in access_matrices_list:  # create multiple readAccess
                    read_accesses_list.append(ReadAccess(buffer=buffer, access_pattern=matrix, computation=computation))

            elif stencil_type == 'square':  # square stencil over all dimensions {2dims (seidel2d)}
                access_matrices_list = [access_matrix]
                for i in range(len(buffer.defining_iterators)):
                    for matrix in access_matrices_list[:]:  # creating a copy to avoid infinite loop
                        access_matrix_copy = matrix.copy()
                        access_matrix_copy[i, -1] = 1
                        access_matrices_list.append(access_matrix_copy)
                        access_matrix_copy = matrix.copy()
                        access_matrix_copy[i, -1] = -1
                        access_matrices_list.append(access_matrix_copy)
                for matrix in access_matrices_list:  # create multiple readAccess
                    read_accesses_list.append(ReadAccess(buffer=buffer, access_pattern=matrix, computation=computation))

            else: #stencil_type == 'uni_dim'   # 1dim stencil on a random dimension over 3points (jacobi1dm, adi, fdtd2d)
                stencil_dim = local_random.randint(0, len(buffer.defining_iterators)-1)
                access_matrices_list = [access_matrix]
                access_matrix_copy = access_matrix.copy()
                access_matrix_copy[stencil_dim, -1] = 1
                access_matrices_list.append(access_matrix_copy)
                access_matrix_copy = access_matrix.copy()
                access_matrix_copy[stencil_dim, -1] = -1
                access_matrices_list.append(access_matrix_copy)

                for matrix in access_matrices_list:  # create multiple readAccess
                    read_accesses_list.append(ReadAccess(buffer=buffer, access_pattern=matrix, computation=computation))

        elif local_random.random() < 0.10:  # probability of making a partial stencil, i.e. one +1 or +2 to one of the dimensions
            access_matrix[local_random.randint(0, access_matrix.shape[0] - 1), -1] = local_random.choice([-1, 1])
            read_accesses_list.append(ReadAccess(buffer=buffer, access_pattern=access_matrix, computation=computation))

        else:  # No stencil
            read_accesses_list.append(ReadAccess(buffer=buffer, access_pattern=access_matrix, computation=computation))

        # TODO add a chance for duplicating some randomly chosen memory accesses

    return read_accesses_list


def choose_expression(computation, read_access_list):  # TODO format of expression not yet defined, using string temporarily
    # Meta-syntax
    # S-> E
    # E-> E o E | (E o E) | (E o V) | memRead
    # V-> Scalar
    # o-> operator
    nb_read_accesses = len(read_access_list)
    operators_list = [' + ', ' - ', '*', '/']
    # nb_operands = local_random.randint(nb_read_accesses, round(nb_read_accesses*1.5)+1)  # chooses the number of operands in the expression
    if nb_read_accesses > 1:
        expression_list = ['E']  # we format expression as a list of characters for convenience, will be transformed to string later. 'E' is a place holder for operands in the expression, and 'V' for scalar values
    elif nb_read_accesses == 1:  # special case when we only  one read access is selected, we give it a chance to expression to contains scalars
        operator = local_random.choices(operators_list, weights=[0.50, 0.25, 0.50, 0.20], k=1)[0]
        if read_access_list[0].buffer == read_access_list[0].computation.write_access.buffer and read_access_list[0].access_pattern_is_simple:  # if the only read buffer is the computation's write buffer and the access_pattern is simple
            expression_list = ['(', 'E', operator, 'V', ')']  # to avoid copying buffer into itself, which results in an omitted computation
        else:
            expression_list = local_random.choices([['E'], ['(', 'E', operator, 'V', ')']], weights=[0.75, 0.25], k=1)[0]
    else:  # nb_read_accesses == 0
        expression_list = ['V']  # if no read_access, just put a scalar

    while expression_list.count('E') < nb_read_accesses:
        indices = [i for i, x in enumerate(expression_list) if x == 'E']  # find the index for each occurrence
        i = local_random.choice(indices)
        operator = local_random.choices(operators_list, weights=[0.50, 0.25, 0.50, 0.20], k=1)[0]
        expression_list[i:i+1] = local_random.choices([['E', operator, 'E'], ['(', 'E', operator, 'E', ')'],
                                                 ['(', 'E', operator, 'V', ')']], weights=[0.75, 0.25, 0.30], k=1)[0]

    # nb_operands = expression_list.count('E')
    # scalar_indices = [i for i, x in enumerate(expression_list) if x == 'E' and (expression_list[i+1] not in operators_list)]  # since in tiramisu scalars can't be before operators
    operand_indices = [i for i, x in enumerate(expression_list) if x == 'E']  # indices where to place read accesses
    # scalar_indices = local_random.choices(scalar_indices, k=nb_operands-len(read_access_list))  # choose the position for scalar values
    for read_access in read_access_list:  # place read_accesses in random operand place holders
        i = local_random.choice(operand_indices)
        expression_list[i] = read_access.write()
        operand_indices.remove(i)

    scalar_indices = [i for i, x in enumerate(expression_list) if x == 'V']  # indices where to place read accesses
    for i in scalar_indices:  # replace the remaining operand placeholders with scalars
        expression_list[i] = str(round(local_random.uniform(0.1, 6), 2))

    expression = ''.join(expression_list)  # transform the list to a string

    expression = simplify_expression(expression)  # simplify the expression
    return expression


# def f(x):
#     # does not need to be normalized
#     return np.exp(-x**2) * np.cos(3*x)**2 * (x-1)**4/np.cosh(1*x)
#
# # def nb_child_loops_dist(x):
# #     return normal(mu, sigma, x)
#
# def sample_dist(g):
#     x = np.linspace(-5,5,1e5)
#     y = g(x)                        # probability density function, pdf
#     cdf_y = np.cumsum(y)            # cumulative distribution function, cdf
#     cdf_y = cdf_y/cdf_y.max()       # takes care of normalizing cdf to 1.0
#     inverse_cdf = interpolate.interp1d(cdf_y, x)    # this is a function
#     return inverse_cdf
#
def sorted_sample(lst, k):
    return [lst[i] for i in sorted(local_random.sample(range(len(lst)), k))]


def random_truncnorm(mu, sigma, low=None, up=None):
    if low is None:
        low = -np.inf
    if up is None:
        up = np.inf
    while True:
        r = local_random.gauss(mu=mu, sigma=sigma)
        if low <= r <= up:
            break
    return r


def simplify_expression(expr):
    expr = simplify(expr)
    expr = expr.replace(lambda x: x.is_Pow and x.exp > 0, lambda x: Symbol('*'.join([str(x.base)]*x.exp)))  # replace Power operator with multiplication
    # expr = expr.replace(lambda x: x.is_Pow and (x.exp < -1), lambda x: Symbol('(1/('+'*'.join([str(x.base)]*(-x.exp))+'))'))  # proposed solution for negative power, ignored for now since it can make the expression no longer simplied
    expr = str(expr)
    # round all scalar values to 3 decimals just for readability
    # nb_float_matches = len(list(re.finditer(r'\b\d+\.?\d*', expr)))
    nb_float_matches = len(list(re.finditer(r'\b\d+\.\d+', expr)))
    for i in range(nb_float_matches):
        # match = list(re.finditer(r'\b\d+\.?\d*', expr))[i]
        match = list(re.finditer(r'\b\d+\.\d+', expr))[i]
        val = match.group()
        val = '{:.3f}'.format(float(val))
        expr = expr[:match.start()] + val + expr[match.end():]

    # find all operations of type cst{*/+-}expr() and cast the cst to an expr(cst)
    while re.findall(r'(\d+\.?\d*)(?:\*|\/|\+|\-)', expr):  # while there are matches
        match = list(re.finditer(r'(\d+\.?\d*)(?:\*|\/|\+|\-)', expr))[0]  # get the first match
        val = match.groups()[0]
        expr = expr[:match.start()] + 'expr(' + '{:.3f}'.format(float(val)) + ')' + expr[match.end() - 1:]

    if expr.startswith('-'):  # workaround for marking the presence of minus in expression
        expr = 'expr(0.0) - ' + expr[1:]
    if expr.startswith('(-'):
        expr = '(expr(0.0) - ' + expr[2:]

    return expr


def has_skippable_loop_multi_comp(prog):
    loop_list = []
    # write_accesses = set()
    # read_accesses = set()
    all_accesses = set()
    for comp in prog.ordered_computations_list[::-1]:
        for lp in [loop.name for loop in comp.parent_iterators_list][::-1]:  # loops should be ordered from the last to be incremented to the first
            if lp not in loop_list:
                loop_list.append(lp)
        # loop_list.update([loop.name for loop in comp.parent_iterators_list])
        all_accesses.update([read_acc.write_buffer_access() for read_acc in comp.expression.read_access_list])
        all_accesses.add(comp.write_access.write_buffer_access())
    # loop_vals = {loop_name:local_random2.randint(1, 100) for loop_name in loop_list}
    # write_accesses_vals = {write_acc: local_random2.randint(1, 100) for write_acc in write_accesses}
    # read_accesses_vals = {read_acc: write_accesses_vals[read_acc] if read_acc in write_accesses_vals else local_random2.randint(1, 100) for read_acc in read_accesses}
    all_accesses_vals = {acc: local_random2.randint(2, 200) for acc in all_accesses}
    # compute written vals at {i0, i1, ..., in}
    original_written_vals = dict()
    for comp in prog.ordered_computations_list:
        subbed_expr = comp.expression.expression[:].replace(' ','')
        # replace input name with their buffer names in expression
        for input in prog.input_list:
            subbed_expr = subbed_expr.replace(input.name, input.buffer.name)
        for acc in all_accesses:
            subbed_expr = subbed_expr.replace(acc, str(all_accesses_vals[acc]))
        subbed_expr = subbed_expr.replace('expr', '')
        try:
            original_written_vals[comp.write_access.write_buffer_access()] = eval(subbed_expr)
        except:  # a division by zero or a simplification error can happen here, just set a random value
            original_written_vals[comp.write_access.write_buffer_access()] = local_random2.randint(2, 200)

        all_accesses_vals[comp.write_access.write_buffer_access()] = original_written_vals[comp.write_access.write_buffer_access()]  # add/update the written value
    # print('orig  ',original_written_vals)
    for loop in loop_list:
        loop_incremented = loop+'+1'
        # add random values to new accesses
        for acc in all_accesses:
            acc = acc.replace(loop, loop_incremented)
            all_accesses_vals[acc] = all_accesses_vals.get(acc, local_random2.randint(2, 200)) # if doesn't exit, create new value
        original_all_accesses_vals = copy.deepcopy(all_accesses_vals)
        new_written_vals = dict()
        for comp in prog.ordered_computations_list:
            if loop not in [l.name for l in comp.parent_iterators_list]: # computation not concerned by the loop
                continue
            subbed_expr = comp.expression.expression[:].replace(' ', '')
            # replace input name with their buffer names in expression
            for input in prog.input_list:
                subbed_expr = subbed_expr.replace(input.name, input.buffer.name)
            subbed_expr = subbed_expr.replace(loop, loop_incremented)
            for acc in all_accesses_vals:
                subbed_expr = subbed_expr.replace(acc, str(all_accesses_vals[acc]))
            subbed_expr = subbed_expr.replace('expr', '')
            write_acc = comp.write_access.write_buffer_access().replace(loop, loop_incremented)
            try:
                new_written_vals[write_acc] = eval(subbed_expr)
            except:  # a division by zero or a simplification error can happen here, just set a random value
                new_written_vals[write_acc] = local_random2.randint(2, 200)
            all_accesses_vals[write_acc] = new_written_vals[write_acc]  # add/update the written value
        # print('new  ',loop, new_written_vals)
        for acc in new_written_vals:
            if acc in original_written_vals:
                if new_written_vals[acc] == original_written_vals[acc]: #writes the same thing in the same place
                    # print(acc, loop, 'is skippable')
                    return True
            if new_written_vals[acc] == original_all_accesses_vals[acc]: # mem cell content hasn't changed after writing (a[i,j]<-a[i,j])
                # print(acc, 'mem value hasnt changed')
                return True

    return False


def has_overwritten_comp(prog):
    writes_dict = dict()
    for comp in prog.ordered_computations_list:
        write_acc_buff = comp.write_access.write_buffer_access()
        if write_acc_buff in writes_dict:
            prev_writing_comp = writes_dict[write_acc_buff]
            if write_acc_buff not in [read_acc.write_buffer_access() for read_acc in comp.expression.read_access_list]:  # if the comp doesn't use the value before overwriting it
                return True
        # if write_acc not overwriting or if it is overwriting but after reading the value, just update the writes_dict
        writes_dict[write_acc_buff] = comp

    return False


