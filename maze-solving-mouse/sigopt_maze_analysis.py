import numpy
from timeit import default_timer

from maze_tools import RightTurnSolver

from sigopt.interface import Connection
SIGOPT_API_TOKEN = 'FIND_YOUR_API_TOKEN_AT_THE_SIGOPT_WEBSITE'
# Determines how you prefer to view your results: True is [sd, mean] and False is [mean, sd]
PREFER_MEAN_ON_Y_AXIS = True


def generate_halton_points(num_points, domain_bounds, skip=0, leap=1, shift=0):
    """Create points sampled from a Halton low discrepancy sequence.

    This can be used to create quasi-random samples from a d-dimensional cube.
    First it creates the desired Halton sequence (subject to skip/leap/shift) in [0, 1]^d and
        then scales and translates it to the domain described in domain_bounds

    skip, leap and shift are more advanced tools for creating alternate low discrepancy sequences.
        skip - omit the first "skip" many terms in the sequence
        leap - use every "leap"th element in the sequence
        shift - a term to set the point (0, 0) in the Halton sequence to be another point
                this can be a number or a (1, d) shape array

    :param num_points: object representing the optimization method to be multistarted
    :type num_points: number of desired points
    :param domain_bounds: boundaries of the d-dimensional cube - the kth dimension is the kth row
    :type domain_bounds: array-like of shape (d, 2)
    :param skip: how many initial Halton points to skip
    :type skip: int >= 0
    :param leap: how many Halton points to leap over when choosing the next in the sequence
    :type leap: int >= 1
    :param shift: where to place the points (0, 0) in the domain [0, 1]^d on which the base Halton sequence is defined
    :type shift: broadcastable to (n, d): array of shape (1, d) or a number
    :return Halton sequence of d-dimensional points, the ith of which appears in the ith row
    :rtype numpy.array of shape (n, d)

    """
    list_of_small_prime_numbers = [2,     3,     5,     7,     11,    13,    17,    19,    23,    29]
    if num_points is None or num_points == 0:
        return numpy.array([])

    dim = len(domain_bounds)
    max_dim = len(list_of_small_prime_numbers)
    if dim > max_dim:
        raise AttributeError('Only {0} dimensions can be considered with the Halton sequence'.format(max_dim))
    if any([db[1] <= db[0] for db in domain_bounds]):
        raise ValueError('Domain minima must be less than maxima: {0}'.format(domain_bounds))

    sequence = skip + leap * numpy.array(range(num_points))
    base = numpy.array(list_of_small_prime_numbers[:dim])[:, numpy.newaxis]

    digs = int(numpy.log1p(numpy.max(sequence)) / numpy.log(numpy.min(base[:, 0])) + 1)
    coefs = numpy.mod(sequence[:, numpy.newaxis, numpy.newaxis] / numpy.power(base, numpy.array(range(digs))), base)
    pts01 = numpy.mod(shift + numpy.sum(coefs * numpy.power(base, -1.0 * numpy.array(range(1, digs + 1))), axis=2), 1)

    pts_scale = numpy.array([db[1] - db[0] for db in domain_bounds])
    pts_min = numpy.array([db[0] for db in domain_bounds])
    return pts_min + pts_scale * pts01


def find_pareto_efficient_mean_sd_vecs(mean_vec, sd_vec, *args, **kwargs):
    """Given mean and standard deviation (sd) vectors, find the [max mean, min sd] efficient points.

    This just cycles through all the mean/sd pairs given and separates the efficient and non-efficient points.
    The efficient points lie on the Pareto frontier and the non-efficient points lie elsewhere in the feasible region.
    The mean and sd points should correspond, i.e., the kth point in each vector occurred with the same parameters.

    In addition to the mean and standard deviation, other information about each observation can be stored in n length
    arrays in the args.  If this is done, then those parameters associated with the Pareto efficient results will be
    returned in the concurrent order.

    :param mean_vec: all mean points that have been observed
    :type mean_vec: array-like of size n
    :param sd_vec: all sd points that have been observed
    :type sd_vec: array_like of size n
    :param args: other vectors that are associated with the kth observation (such as the parameters)
    :type args: each arg should be an array-like of size n
    :param kwargs: mean_on_y_axis - the only acceptable kwarg, says whether mean or sd should be on the y-axis
    :type kwargs: mean_on_y_axis - bool (default is True)
    :return: If args is not passed, two 2d arrays: the points on the frontier, and those that are non-efficient
             If args is passed, a matrix is also returned which contains the args associated with the Pareto frontier
             NOTE: If mean_on_y_axis=False, then the returned values are [mean, sd], not [sd, mean]
    """
    assert len(mean_vec) == len(sd_vec)
    assert all([len(a) == len(mean_vec) for a in args])
    mean_on_y_axis = kwargs.get('mean_on_y_axis', PREFER_MEAN_ON_Y_AXIS)

    pareto_efficient_parameters = []
    good_ind = []
    bad_ind = []
    objpred = numpy.array([sd_vec, -mean_vec]) if mean_on_y_axis else numpy.array([-mean_vec, sd_vec])
    for j, o in enumerate(objpred.T):
        if numpy.all(numpy.any(o[:, numpy.newaxis] <= objpred, axis=0)):
            good_ind.append(j)
            if args:
                pareto_efficient_parameters.append([a[j] for a in args])
        else:
            bad_ind.append(j)
    pareto_front = (numpy.array([[1], [-1]]) if mean_on_y_axis else numpy.array([[-1], [1]])) * objpred[:, good_ind]
    non_pareto_front = (numpy.array([[1], [-1]]) if mean_on_y_axis else numpy.array([[-1], [1]])) * objpred[:, bad_ind]

    if args:
        return pareto_front.T, non_pareto_front.T, numpy.array(pareto_efficient_parameters)
    return pareto_front.T, non_pareto_front.T


# This is for the same purpose as the find_pareto_efficient_mean_sd_vecs function but is instead passed matrices of
# mean and standard deviation values (and possibly also args).
# This can be convenient if the mats already exist for plotting purposes.
def find_pareto_efficient_mean_sd_mat(mean_mat, sd_mat, *args, **kwargs):
    assert mean_mat.shape == sd_mat.shape
    assert all([a.shape == mean_mat.shape for a in args])
    mean_vec = mean_mat.reshape(-1)
    sd_vec = sd_mat.reshape(-1)
    mean_on_y_axis = kwargs.get('mean_on_y_axis', PREFER_MEAN_ON_Y_AXIS)

    if args:
        parameter_vecs = [a.reshape(-1) for a in args]
        return find_pareto_efficient_mean_sd_vecs(mean_vec, sd_vec, *parameter_vecs, mean_on_y_axis=mean_on_y_axis)
    return find_pareto_efficient_mean_sd_vecs(mean_vec, sd_vec, mean_on_y_axis=mean_on_y_axis)


def plot_pareto_frontier(pareto_front, non_pareto_front, mean_on_y_axis=PREFER_MEAN_ON_Y_AXIS, fontsize=20):
    """Takes in the results of the Pareto efficient points found above and plots them.

    The ordering of the data needs to match mean_on_y_axis.  So if mean_on_y_axis is true, then pareto_front should
        contain the values on the Pareto frontier in the form [sd, mean] and if mean_on_y_axis == False, then the data
        should be passed as [mean, sd].

    :param pareto_front: 2d array containing the points on the Pareto frontier
    :type pareto_front: numpy.array of shape (m, 2)
    :param non_pareto_front: 2d array containing the points not on the Pareto frontier
    :type non_pareto_front: numpy.array of shape (n, 2)
    :param mean_on_y_axis: Flag to say how the data is organized
    :type mean_on_y_axis: bool
    :param fontsize: Size of the text for matplotlib axis labels
    :type fontsize: int > 0
    """
    from matplotlib import pyplot as plt
    plt.plot(non_pareto_front[:, 0], non_pareto_front[:, 1], 'o', color='#1F407D')
    plt.plot(pareto_front[:, 0], pareto_front[:, 1], 'o', color='#E84557')
    if mean_on_y_axis:
        plt.xlabel('escape time SD', fontsize=fontsize)
        plt.ylabel('escape time mean', fontsize=fontsize)
    else:
        plt.ylabel('escape time SD', fontsize=fontsize)
        plt.xlabel('escape time mean', fontsize=fontsize)


def plot_hist_prop(data, plot_it=False, **histogram_kwargs):
    """Create histogram data (and maybe plot it) which is normalize to be an empirical mass distribution.

    :param data: all the relevant solving times that we have observed the mouse take
    :type data: list-like
    :param plot_it: should this function also plot the histogram
    :type plot_it: bool
    :param histogram_kwargs: anything you want passed directly to numpy.histogram function
    :type histogram_kwargs: kwargs-style
    :return the locations, (normalized) heights and widths of the histogram components
    :rtype 3 numpy.arrays
    """
    hist, bins = numpy.histogram(data, **histogram_kwargs)
    hist = numpy.array(hist, dtype=float) / sum(hist)
    widths = numpy.diff(bins)

    bins = bins[:-1]
    if plot_it:
        from matplotlib import pyplot as plt
        plt.bar(bins, hist, widths)
        plt.xlim((0, 2))  # The mouse can never take more than double the number of cells to solve the maze
        plt.ylabel('proportion')
        plt.xlabel('search duration')
    return bins, hist, widths


class SigOptMazeFrontierSolver(object):
    """
    This is an object which allows for the study of random parameters on the maze construction/solution problem.  It
    allows for the left, up and right construction probabilities to be manipulated in some domain, while the down
    probability is held constant.
    """

    def __init__(self, api_token=None, parameter_domain=None, maze_size=(30, 30), num_tests=100, down_prob=1.0):
        """Create a new SigOpt testing tool.

        The api_token is a long string which can be found on your account at www.sigopt.com/tokens/info.
            You can either pass that string here or you can modify this file to save it on your local machine.

        parameter_domain describes the allowable values for the relative probabilities of constructing the maze with
            "left", "up" or "right" moves.  These probabilities are relative to the probability of moving "down" which
            is always fixed at 1.0 (by default).  The domain should be a numpy.array of shape (3, 2), where:
                * the first row denotes the lower and upper bound for the "left" domain,
                * the second row denotes the lower and upper bound for the "up" domain,
                * the third row denotes the lower and upper bound for the "right" domain,

        :param api_token: The API token used to access SigOpt (or SIGOPT_API_TOKEN can be modified above)
        :type api_token: str
        :param parameter_domain: Domain on which the solver should consider the three parameters
        :type parameter_domain: numpy.array of shape (3, 2)
        :param maze_size: size of the maze to randomly generate and test (must be at least 2 rows and 2 columns)
        :type maze_size: tuple of length 2
        :param num_tests: How many random mazes should be constructed to estimate the mean and standard deviation
        :type num_tests: int > 1, (will default to the value stored at construction)
        :param down_prob: The relative probability of moving down (against which other probabilities are compared)
        :type down_prob: float > 0
        """
        self.conn = Connection(client_token=api_token or SIGOPT_API_TOKEN)
        self.experiment = None
        self.domain = parameter_domain or numpy.array([[.01, 100]] * 3)
        if not numpy.all(self.domain[:, 1] > self.domain[:, 0]):
            raise AssertionError('The minimum values (column 1) in the domain must be less than the maximum (column 2)')
        assert type(maze_size) in (list, tuple) and len(maze_size) == 2 and all([m > 1 for m in maze_size])
        self.maze_size = maze_size
        assert type(num_tests) == int and num_tests > 1
        self.num_tests = num_tests
        assert down_prob > 0
        self.down_prob = float(down_prob)

    def _execute_assignments(self, num_tests, l, u, r):
        """Estimate the mean and standard deviation for given left, up and right turn relative probabilities"""
        num_points_searched = []
        rts = RightTurnSolver()
        for _ in range(num_tests):
            rts.generate_random_maze(*self.maze_size, prob=(l, u, r, self.down_prob))
            rts.solve()
            num_points_searched.append(len(rts.path) / float(numpy.prod(rts.shape)))
        return numpy.mean(num_points_searched), numpy.sqrt(numpy.var(num_points_searched))

    def execute_low_discrepancy_testing(
            self,
            num_points,
            num_tests=None,
            log_sample=True,
            verbose=False,
            halton_kwargs=None
    ):
        """Run a low discrepancy sampling over the domain to generate an understanding of the feasible region.

        :param num_points: The number of points at which to sample
        :type num_points: int > 0
        :param num_tests: How many random mazes should be constructed to estimate the mean and standard deviation
        :type num_tests: int > 1, (will default to the value stored at construction)
        :param log_sample: Should this low discrepancy sampling occur on a logarithmically scaled version of the domain
        :type log_sample: bool
        :param verbose: A function to call to receive the current iteration and determine if progress should be output
        :type verbose: callable or bool (if bool and True will be output after each tenth test)
        :param halton_kwargs: kwargs to pass to the generate_halton_points function
        :type halton_kwargs: dictionary (defaults to {'shift': numpy.random.random()})
        :return: Arrays recording the mean, standard deviation results, and the left, up and right values tested
        :rtype: 5 numpy.array objects
        """
        num_tests = num_tests or self.num_tests
        verbose = (lambda it: it % 10 == 0) if verbose is True else verbose
        assert verbose in (False, None) or callable(verbose)
        halton_kwargs = {} if halton_kwargs is None else {'shift': numpy.random.random()}
        if log_sample:
            test_points = numpy.exp(generate_halton_points(num_points, numpy.log(self.domain), **halton_kwargs))
        else:
            test_points = generate_halton_points(num_points, self.domain, **halton_kwargs)

        mean_vec = numpy.empty(num_points)
        sd_vec = numpy.empty(num_points)
        start = default_timer()
        for k, test_point in enumerate(test_points):
            mean_vec[k], sd_vec[k] = self._execute_assignments(num_tests, *test_point)
            if verbose and verbose(k):
                end = default_timer()
                print '{0:5d} iterations completed, {1:4.1f} seconds since last report'.format(k, end - start)
                start = end

        left_vec, up_vec, right_vec = test_points.T
        return mean_vec, sd_vec, left_vec, up_vec, right_vec

    def create_sigopt_experiment(self, name=None):
        self.experiment = self.conn.experiments().create(
            name=name or 'Scalarized Mouse Maze Solver',
            parameters=[
                {'name': 'left_prob_log', 'bounds': {'max': numpy.log(100), 'min': numpy.log(0.01)}, 'type': 'double'},
                {'name': 'up_prob_log', 'bounds': {'max': numpy.log(100), 'min': numpy.log(0.01)}, 'type': 'double'},
                {'name': 'right_prob_log', 'bounds': {'max': numpy.log(100), 'min': numpy.log(0.01)}, 'type': 'double'},
            ],
        )

    def _extract_data_from_experiment(self, experiment_id=None):
        experiment_id = experiment_id or self.experiment.id
        num_observations = self.conn.experiments(experiment_id).fetch().progress.observation_count
        mean_vec = numpy.empty(num_observations)
        sd_vec = numpy.empty(num_observations)
        right_vec = numpy.empty(num_observations)
        up_vec = numpy.empty(num_observations)
        left_vec = numpy.empty(num_observations)

        for k, observation in enumerate(self.conn.experiments(experiment_id).observations().fetch().iterate_pages()):
            mean_vec[k] = observation.metadata['mean']
            sd_vec[k] = observation.metadata['std_dev']
            left_vec[k] = numpy.exp(observation.assignments['left_prob_log'])
            up_vec[k] = numpy.exp(observation.assignments['up_prob_log'])
            right_vec[k] = numpy.exp(observation.assignments['right_prob_log'])
        return mean_vec, sd_vec, left_vec, up_vec, right_vec


class SigOptMazeWeightedSumFrontierSolver(SigOptMazeFrontierSolver):
    def __init__(self, api_token=None, parameter_domain=None, maze_size=(30, 30), num_tests=100, down_prob=1.0):
        super(SigOptMazeWeightedSumFrontierSolver, self).__init__(
            api_token=api_token,
            parameter_domain=parameter_domain,
            maze_size=maze_size,
            num_tests=num_tests,
            down_prob=down_prob,
        )

    @staticmethod
    def weighted_sum_scalarization(mean_weight, mean, sd):
        return mean_weight * mean - (1 - mean_weight) * sd

    def _form_sigopt_weighted_sum_observation_from_suggestion(self, num_tests, suggestion, mean_weight):
        log_assignments = suggestion.assignments
        l = numpy.exp(log_assignments['left_prob_log'])
        u = numpy.exp(log_assignments['up_prob_log'])
        r = numpy.exp(log_assignments['right_prob_log'])
        m, s = self._execute_assignments(num_tests, l, u, r)
        return {
            'suggestion': suggestion.id,
            'value': self.weighted_sum_scalarization(mean_weight, m, s),
            'metadata': {'mean_weight': mean_weight, 'mean': m, 'std_dev': s},
        }

    def _add_weighted_sum_historical_info(self, previous_experiment_id, mean_weight):
        if previous_experiment_id is None:
            return
        for observation in self.conn.experiments(previous_experiment_id).observations().fetch().iterate_pages():
            new_metadata = observation.metadata
            new_metadata['mean_weight'] = mean_weight
            log_assignments = observation.assignments
            self.conn.experiments(self.experiment.id).observations().create(
                assignments=log_assignments,
                value=self.weighted_sum_scalarization(mean_weight, new_metadata['mean'], new_metadata['std_dev']),
                metadata=new_metadata,
            )

    def execute_sigopt_weighted_sum_scalarized_optimization(self, num_evals, num_tests, mean_weight):
        """This runs a SigOpt optimization over a specific weighted sum scalarized form of the multicriteria problem.

        The scalar objective which is then maximized is found in self.weighted_sum_scalarization.  It is just a linear
            combination of the mean and -sd with weights that sum to 1.

        :param num_evals: The number of suggestion/observation iterations to execute with SigOpt
        :type num_evals: int > 0
        :param num_tests: How many random mazes should be constructed to estimate the mean and standard deviation
        :type num_tests: int > 1, (will default to the value stored at construction)
        :param mean_weight: The weight that will be applied to the mean term in the scalarization
        :type mean_weight: float within (0, 1)
        """
        for col in range(num_evals):
            suggestion = self.conn.experiments(self.experiment.id).suggestions().create()
            observation = self._form_sigopt_weighted_sum_observation_from_suggestion(num_tests, suggestion, mean_weight)
            self.conn.experiments(self.experiment.id).observations().create(**observation)

    def execute_sigopt_weighted_sum_frontier_search(self, exp_dict_list, verbose):
        """Conduct a sequence of weighted sum scalarized optimizations to approximated the Pareto frontier.

        NOTE: Executing this will create len(exp_dict_list) SigOpt experiments!  If you are modifying this code or
            experimenting on a problem of your own, consider using the development API token to not count against your
            experiment limit during debugging.  See https://sigopt.com/docs/overview/authentication for information.

        The exp_dict_list object is a list containing dictionaries with the instructions as to how the sequence
            of experiments should operate.  An example would be:
            exp_dict_list = [
                {'mean_weight': .1, 'num_evals': 30, 'num_tests': 50},
                {'mean_weight': .5, 'num_evals': 20, 'num_tests': 80},
                ...
            ]
        In this case, the second experiment to be run would have scalarization weight .5 for the mean, 20 SigOpt
            iterations, and 80 maze generation/solutions to estimate the observed mean and standard deviation.  All of
            the entries in exp_dict_list will be run in the order in which they appear.

        To take advantage of any available previous experiments, earlier results are used to seed the current experiment
            before executing the scalarized optimization.  We do this by storing the mean and variance information used
            to generate the scalarized objective as metadata and then extracting that information from the previous
            SigOpt experiment.

        :param exp_dict_list: list containing entries explaining the experiments to be run
        :type exp_dict_list: list
        :param verbose: Output progress after each SigOpt optimization completes
        :type verbose: bool
        :return: Arrays recording the mean, standard deviation results, and the left, up and right values tested
        :rtype: 5 numpy.array objects
        """
        assert type(exp_dict_list) in (list, tuple)
        for ed in exp_dict_list:
            assert type(ed) == dict
            assert set(ed.keys()) == {'mean_weight', 'num_evals', 'num_tests'}
            assert type(ed['mean_weight']) == float and 0 < ed['mean_weight'] < 1
            assert type(ed['num_evals']) == int and ed['num_evals'] > 0
            assert type(ed['num_tests']) == int and ed['num_tests'] > 0
        # Note(Mike) - Patrick, I don't know which of these looks better, or whether this is appropriate at all
        # assert type(exp_dict_list) in (list, tuple) and all([type(ed) == dict for ed in exp_dict_list])
        # assert all([set(ed.keys()) == {'mean_weight', 'num_evals', 'num_tests'} for ed in exp_dict_list])
        # assert all([type(ed['mean_weight']) == float and 0 < ed['mean_weight'] < 1 for ed in exp_dict_list])
        # assert all([type(ed['num_evals']) == int and ed['num_evals'] > 0 for ed in exp_dict_list])
        # assert all([type(ed['num_tests']) == int and ed['num_tests'] > 0 for ed in exp_dict_list])

        previous_sigopt_experiment_id = None
        for k, experiment_info in enumerate(exp_dict_list):
            self.create_sigopt_experiment(name='Weighted sum with weight={0}'.format(experiment_info['mean_weight']))
            self._add_weighted_sum_historical_info(previous_sigopt_experiment_id, experiment_info['mean_weight'])
            self.execute_sigopt_weighted_sum_scalarized_optimization(**experiment_info)
            previous_sigopt_experiment_id = self.experiment.id
            if verbose:
                print 'Experiment {0}: id {1}, Info {2}'.format(k, previous_sigopt_experiment_id, experiment_info)

        return self._extract_data_from_experiment()


class SigOptMazeConstraintFrontierSolver(SigOptMazeFrontierSolver):
    def __init__(self, api_token=None, parameter_domain=None, maze_size=(30, 30), num_tests=100, down_prob=1.0):
        super(SigOptMazeConstraintFrontierSolver, self).__init__(
            api_token=api_token,
            parameter_domain=parameter_domain,
            maze_size=maze_size,
            num_tests=num_tests,
            down_prob=down_prob,
        )

    @staticmethod
    def _form_constraint_observation_update(mean_constraint, mean, std_dev):
        observation = {'metadata': {'mean_constraint': mean_constraint, 'mean': mean, 'std_dev': std_dev}}
        if mean > mean_constraint:
            observation.update({'value': -std_dev})
        else:
            observation.update({'failed': True})
        return observation

    def _form_sigopt_constraint_observation_from_suggestion(self, num_tests, suggestion, mean_constraint):
        log_assignments = suggestion.assignments
        l = numpy.exp(log_assignments['left_prob_log'])
        u = numpy.exp(log_assignments['up_prob_log'])
        r = numpy.exp(log_assignments['right_prob_log'])
        m, s = self._execute_assignments(num_tests, l, u, r)
        observation = {'suggestion': suggestion.id}
        observation.update(self._form_constraint_observation_update(mean_constraint, m, s))
        return observation

    def _add_constraint_historical_info(self, previous_experiment_id, mean_constraint):
        if previous_experiment_id is None:
            return
        for observation in self.conn.experiments(previous_experiment_id).observations().fetch().iterate_pages():
            new_metadata = observation.metadata
            new_metadata['mean_constraint'] = mean_constraint
            log_assignments = observation.assignments
            observation = {'assignments': log_assignments}
            observation.update(self._form_constraint_observation_update(**new_metadata))
            self.conn.experiments(self.experiment.id).observations().create(**observation)

    def execute_sigopt_constraint_scalarized_optimization(self, num_evals, num_tests, mean_constraint):
        """This runs a SigOpt optimization over a specific constraint scalarized form of the multicriteria problem.

        The scalar objective which is then maximized is encapsulated in self._form_constraint_observation_update.
            If the mean is greater than the desired constraint, then the negative of the standard deviation is reported
            to SigOpt (to minimize the standard deviation).  If the mean is not beyond the desired constraint, the
            observation is reported as a failure.

        :param num_evals: The number of suggestion/observation iterations to execute with SigOpt
        :type num_evals: int > 0
        :param num_tests: How many random mazes should be constructed to estimate the mean and standard deviation
        :type num_tests: int > 1, (will default to the value stored at construction)
        :param mean_constraint: The lower limit of acceptable means; sd with too low a mean register as failures
        :type mean_constraint: float > 0
        """
        for col in range(num_evals):
            suggestion = self.conn.experiments(self.experiment.id).suggestions().create()
            observation = self._form_sigopt_constraint_observation_from_suggestion(
                num_tests,
                suggestion,
                mean_constraint,
            )
            self.conn.experiments(self.experiment.id).observations().create(**observation)

    def execute_sigopt_constraint_frontier_search(self, exp_dict_list, verbose):
        """Conduct a sequence of constraint-based scalarized optimizations to approximated the Pareto frontier.

        NOTE: Executing this will create len(exp_dict_list) SigOpt experiments!  If you are modifying this code or
            experimenting on a problem of your own, consider using the development API token to not count against your
            experiment limit during debugging.  See https://sigopt.com/docs/overview/authentication for information.

        The exp_dict_list object is a list containing dictionaries with the instructions as to how the sequence
            of experiments should operate.  An example would be:
            exp_dict_list = [
                {'mean_constraint': .9, 'num_evals': 30, 'num_tests': 50},
                {'mean_constraint': .95, 'num_evals': 20, 'num_tests': 80},
                ...
            ]
        In this case, the second experiment to be run would have mean constraint .95 (if the estimated mean is less
            than that the observation is a failure), 20 SigOpt iterations, and 80 maze generation/solutions to estimate
            the observed mean and standard deviation.  All of the entries in exp_dict_list will be run in the order in
            which they appear.

        To take advantage of any available previous experiments, earlier results are used to seed the current experiment
            before executing the scalarized optimization.  We do this by storing the mean and variance information used
            to generate the scalarized objective as metadata and then extracting that information from the previous
            SigOpt experiment.

        Because the failure provides little or no information to SigOpt, it can be difficult to execute this constraint
            based scalarization strategy when you have little information about the multicriteria component function
            ranges.  It can be helpful to run some low discrepancy searches first to gain some initial insight about
            what appropriate constraints are.

        :param exp_dict_list: list containing entries explaining the experiments to be run
        :type exp_dict_list: list
        :param verbose: Output progress after each SigOpt optimization completes
        :type verbose: bool
        :return: Arrays recording the mean, standard deviation results, and the left, up and right values tested
        :rtype: 5 numpy.array objects
        """
        assert type(exp_dict_list) in (list, tuple)
        for ed in exp_dict_list:
            assert type(ed) == dict
            assert set(ed.keys()) == {'mean_constraint', 'num_evals', 'num_tests'}
            assert type(ed['mean_constraint']) == float and ed['mean_constraint'] > 0
            assert type(ed['num_evals']) == int and ed['num_evals'] > 0
            assert type(ed['num_tests']) == int and ed['num_tests'] > 0

        previous_sigopt_experiment_id = None
        for k, experiment_info in enumerate(exp_dict_list):
            self.create_sigopt_experiment(name='Constraint on mean={0}'.format(experiment_info['mean_constraint']))
            self._add_constraint_historical_info(previous_sigopt_experiment_id, experiment_info['mean_constraint'])
            self.execute_sigopt_constraint_scalarized_optimization(**experiment_info)
            previous_sigopt_experiment_id = self.experiment.id
            if verbose:
                print 'Experiment {0}: id {1}, Info {2}'.format(k, previous_sigopt_experiment_id, experiment_info)

        return self._extract_data_from_experiment()
