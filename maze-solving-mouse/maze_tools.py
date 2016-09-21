import numpy
try:  # This can still be used without matplotlib, if no plotting is required
    import matplotlib.pyplot as plt
except ImportError:
    plt = NotImplemented


# Maybe it would be nice to have an import/export json function
class RightTurnSolver(object):
    def __init__(self):
        self.maze = None
        self.path = None
        self.generation_history = None
        self.start = None
        self.end = None

    def set_maze(self, new_maze, start=None, end=None):
        """Allows the user to place a pre-generated maze into this object.

        The maze object must have shape (m, n, 4) where the desired maze has m rows and n columns.  When accessing
            new_maze[i, j, :] the result shows which paths exist.  For example, if new_maze[4, 6, :] was [0, 1, 1, 0]
            then the cell associated with the 4th row and 6th column of the maze should have walls in the left and down
            locations and paths in the up and right locations.  This is because we order [left, up, right, down].

        Note that no checking will be conducted to confirm that the maze is acceptable (solvable).  Also, no checking
            will occur to confirm that there are walls on the boundary.

        :param new_maze: description of the maze
        :type new_maze: numpy.array of shape (m, n, 4)
        :param start: location at which the mouse would start the maze (row, column)
        :type start: numpy.array with one dimension of length 2
        :param end: location at which the mouse woudl end the maze (row, column)
        :type end: numpy.array with one dimension of length 2
        """
        assert type(new_maze) == numpy.ndarray
        assert len(new_maze.shape) == 3
        assert new_maze.shape[2] == 4
        self.maze = numpy.copy(new_maze)
        self.generation_history = None
        self.reset_solver()
        self.start = start or numpy.array([0, 0])
        self.end = end or numpy.array([s - 1 for s in self.shape])

    @property
    def shape(self):
        return (0, 0) if self.maze is None else self.maze.shape[:2]

    def reset_solver(self):
        self.path = None
        self.start = None
        self.end = None

    def generate_random_maze(self, num_rows, num_cols, prob=(.25, .25, .25, .25)):
        """Generate and store a maze in this object.

        Borrowed from https://en.wikipedia.org/wiki/Maze_generation_algorithm
        This will always create a maze which starts at the "top left" and ends at the "bottom right"
        After this is called, there will be a maze stored in self.maze
        Also, the generation history will be stored in self.generation_history, but that is only for plotting and serves
            no practical purpose afterwards

        :param num_rows: how many rows should appear in the maze
        :type num_rows: int > 1
        :param num_cols: how many columns should appear in the maze
        :type num_cols: int > 1
        :param prob: the probability of turning (left, up, right, down) during random maze generation
        :type prob: list-like of length 4
        """

        # The array maze_info is going to hold the array information for each cell.
        # The first four coordinates tell if walls exist on those sides
        # and the fifth indicates if the cell has been visited in the search.
        # maze_info(LEFT, UP, RIGHT, DOWN, CHECK_IF_VISITED)
        maze_info = numpy.zeros((num_rows, num_cols, 5), dtype=int)

        # Set starting row and column
        r = 0
        c = 0
        history = [(r, c)]
        generation_history = []

        assert all([p >= 0 for p in prob])
        assert any([p > 0 for p in prob])
        p_dict = dict(zip(('L', 'U', 'R', 'D'), prob))

        # Normalize probability vector to sum to one for possible values from ('L', 'U', 'R', 'D')
        # The 1e-15 avoids the chance where all probabilities are 0 (degenerate case)
        def normalize_prob(check):
            temp_p_vec = numpy.array([p_dict[direction] for direction in check], dtype=float) + 1e-15
            return temp_p_vec / sum(temp_p_vec)

        # Trace a path though the cells of the maze and open walls along the path.
        # We do this with a while loop, repeating the loop until there is no history,
        # which would mean we backtracked to the start.
        while history:
            generation_history.append((r, c))
            maze_info[r, c, 4] = 1  # designate this location as visited

            # create a list of all possible directions to which the generation tool can move
            check = []
            if c > 0 and maze_info[r, c - 1, 4] == 0:
                check.append('L')
            if r > 0 and maze_info[r - 1, c, 4] == 0:
                check.append('U')
            if c < num_cols - 1 and maze_info[r, c + 1, 4] == 0:
                check.append('R')
            if r < num_rows - 1 and maze_info[r + 1, c, 4] == 0:
                check.append('D')

            # If there is a valid cell to which it can move, choose one of them randomly
            if len(check):
                # Mark the walls between cells as open if we move
                history.append((r, c))
                move_direction = numpy.random.choice(check, p=normalize_prob(check))
                if move_direction == 'L':
                    maze_info[r, c, 0] = 1
                    c -= 1
                    maze_info[r, c, 2] = 1
                if move_direction == 'U':
                    maze_info[r, c, 1] = 1
                    r -= 1
                    maze_info[r, c, 3] = 1
                if move_direction == 'R':
                    maze_info[r, c, 2] = 1
                    c += 1
                    maze_info[r, c, 0] = 1
                if move_direction == 'D':
                    maze_info[r, c, 3] = 1
                    r += 1
                    maze_info[r, c, 1] = 1
            else:  # If there are no valid cells to move to, retrace one step back in history
                r, c = history.pop()

        self.maze = maze_info[:, :, :4]
        self.generation_history = numpy.array(generation_history)
        self.reset_solver()
        self.start = numpy.array([0, 0])
        self.end = numpy.array([s - 1 for s in self.shape])

    def solve(self, start=None, end=None, max_steps=None, verbose=False):
        """Solve the maze that has already been stored (either manually or generated internally).

        self.path is stored after this maze is solved to record where the mouse traveled.  This gives both the info
            for plotting, but also the number of steps it took the mouse.

        max_steps can limit the number of steps to a desired duration (if you want to limit how many steps the mouse
            can take).  It can also be used to prevent the mouse from entering an infinite loop.  This will not be
            possible if the maze was generated by this tool, but could be possible if you have manually entered a maze.

        :param start: starting location (row, column) of the maze, to override what is stored internally
        :type start: numpy.array with one dimension of length 2
        :param end: ending location (row, column) of the maze, to override what is stored internally
        :type end: numpy.array with one dimension of length 2
        :param max_steps: maximum number of steps that the solver may take to solve this problem
        :type max_steps: int > 0
        :param verbose: print out all the steps that the mouse is taking while solving the maze
        :type verbose: bool
        :return Has the maze been solved in an acceptable number of steps
        :rtype bool
        """
        if self.maze is None:
            raise AssertionError('No maze has been set')

        start = start or numpy.array([0, 0])
        end = end or numpy.array([s - 1 for s in self.shape])
        max_steps = max_steps or numpy.inf

        location = numpy.copy(start)
        self.path = [numpy.copy(location)]
        facing = 2  # Start by facing right: [0, 1, 2, 3] == [left, up, right, down]
        steps = numpy.array([[0, -1], [-1, 0], [0, 1], [1, 0]])
        num_steps = 0
        while any(location != end) and num_steps < max_steps:
            if verbose:
                print 'Visiting {0}'.format(location)
            available_directions = self.maze[location[0], location[1], :]
            moved = False
            while not moved:
                turn_right = numpy.mod(facing + 1, 4)
                turn_left = numpy.mod(facing - 1, 4)
                if available_directions[turn_right]:  # If I can turn right, do so and move
                    facing = turn_right
                    location += steps[facing]
                    moved = True
                elif available_directions[facing]:  # If I can't turn right, true to go forward
                    location += steps[facing]
                    moved = True
                else:  # If can't turn right or go forward, turn left and try again
                    facing = turn_left
            self.path.append(numpy.copy(location))
            num_steps += 1

        self.path = numpy.array(self.path)
        self.start = numpy.copy(start)
        self.end = numpy.copy(end)

        if any(location != end):
            return False
        return True

    """
    Below are all the plotting tools which are only useful for generating the content in the blog post
    """

    def _form_maze_image_for_plotting(self, maze=None):
        num_rows, num_cols = self.shape
        image = numpy.zeros((num_rows * 10, num_cols * 10), dtype=numpy.uint8)
        maze = self.maze if maze is None else maze
        # The array image is going to be the output image to display
        for row in range(0, num_rows):
            for col in range(0, num_cols):
                cell_data = maze[row, col]
                for i in range(10 * row + 1, 10 * row + 9):
                    image[i, range(10 * col + 1, 10 * col + 9)] = 255
                    if cell_data[0] == 1:
                        image[range(10 * row + 1, 10 * row + 9), 10 * col] = 255
                    if cell_data[1] == 1:
                        image[10 * row, range(10 * col + 1, 10 * col + 9)] = 255
                    if cell_data[2] == 1:
                        image[range(10 * row + 1, 10 * row + 9), 10 * col + 9] = 255
                    if cell_data[3] == 1:
                        image[10 * row + 9, range(10 * col + 1, 10 * col + 9)] = 255
        return image

    @staticmethod
    def _pcc(pt):  # stands for plot cell center
        return 10 * pt + 5

    def _plot_maze_alone_from_image(self, image, ax, figsize, markersize):
        start_row, start_col = self._pcc(self.start)
        end_row, end_col = self._pcc(self.end)
        if ax is None:
            _ = plt.figure(figsize=figsize)
            ax = plt.gca()
        ax.imshow(image, cmap=plt.cm.Greys_r, interpolation='none')
        # Note that the column and row get reversed because of the way imshow renders the image
        ax.plot(start_col, start_row, 'sb', markersize=markersize)
        ax.plot(end_col, end_row, '*y', markersize=1.5 * markersize)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((0, image.shape[1] - 1))
        ax.set_ylim((image.shape[0] - 1, 0))
        return ax

    def plot_maze(self, ax=None, figsize=(10, 5), also_solution=False, markersize=20):
        """Create a plot with of the maze, and possibly the solution, using matplotlib.

        :param ax: matplotlib axis object on which to make the plot; if None, will create one
        :type ax: matplotlib.axis
        :param figsize: size of the figure for matplotlib to create (if ax == None)
        :type figsize: list-like of length 2
        :param also_solution: should the solution also be plotted
        :type also_solution: bool
        :param markersize: how large should the start/end/solution points be
        :type markersize: int >= 0
        :return the axis on which this plot exists
        :rtype matplotlib.axis
        """
        if self.maze is None:
            raise ValueError('No maze exists')

        ax = self._plot_maze_alone_from_image(self._form_maze_image_for_plotting(), ax, figsize, markersize)

        if also_solution:
            if self.path is None:
                raise AssertionError('To plot the solution, the solution must be found')
            yp, xp = self._pcc(self.path).T  # Transpose reverses the order
            ax.set_xticks([])
            ax.set_yticks([])
            ax.plot(xp, yp, 'or')
        return ax

    def gif_maze_generation(self, prefix, image_format='png', duration=None, figsize=(10, 5), markersize=20):
        """Save a sequence of figures from which a gif can be formed explaining the random creation of this maze.

        :param prefix: characters to define this family of plots, e.g., gif/happy0.png, gif/happy1.png, ...
        :type prefix: string
        :param image_format: a matplotlib saveable format: probably one of png, pdf, ps, eps, svg
        :type image_format: string
        :param duration: how many points to include in the duration (if it's a really big maze)
        :type duration: int > 0
        :param figsize: size of the figure for matplotlib to create (if ax == None)
        :type figsize: list-like of length 2
        :param markersize: how large should the start/end/solution points be
        :type markersize: int >= 0
        """
        if self.maze is None or self.generation_history is None:
            raise AssertionError('This can only be called after creating a random maze with this object')
        duration = duration or len(self.generation_history)
        partial_maze = numpy.zeros((self.shape[0], self.shape[1], 4), dtype=int)
        for k in range(duration):
            r, c = self.generation_history[k, :]
            partial_maze[r, c, :] = self.maze[r, c, :]
            partial_image = self._form_maze_image_for_plotting(partial_maze)
            self._plot_maze_alone_from_image(partial_image, None, figsize=figsize, markersize=0)

            yp, xp = self._pcc(self.generation_history[:k + 1, :]).T
            plt.plot(xp[-1], yp[-1], 'o', color='#1f407D', markersize=markersize)
            plt.savefig(prefix + '{0}.{1}'.format(k, image_format), format=image_format)

    def gif_solution_plot(self, prefix, image_format='png', figsize=(10, 5), markersize=20):
        """Save a sequence of figures from which a gif can be formed explaining the solution of this maze.

        :param prefix: characters to define this family of plots, e.g., gif/happy0.png, gif/happy1.png, ...
        :type prefix: string
        :param image_format: a matplotlib saveable format: probably one of png, pdf, ps, eps, svg
        :type image_format: string
        :param figsize: size of the figure for matplotlib to create (if ax == None)
        :type figsize: list-like of length 2
        :param markersize: how large should the start/end/solution points be
        :type markersize: int >= 0
        """
        if self.maze is None:
            raise AssertionError('The maze must be stored to make this plot')
        if self.path is None:
            raise AssertionError('To plot the solution, the solution must be found')
        image = self._form_maze_image_for_plotting()
        for k in range(len(self.path)):
            self._plot_maze_alone_from_image(image, None, figsize, markersize)
            yp, xp = self._pcc(self.path[:k + 1, :]).T

            plt.plot(xp, yp, '.k', markersize=markersize / 2)
            plt.plot(xp[-1], yp[-1], 'or', markersize=markersize / 2)
            plt.savefig(prefix + '{0}.{1}'.format(k, image_format), format=image_format)
