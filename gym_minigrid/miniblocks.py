from gym_minigrid.minigrid import *

class Block(WorldObj):
    def __init__(self):
        super().__init__('block', 'red')

    def can_overlap(self):
        return False

    def can_move(self):
        return True

    def render(self, r):
        self._set_color(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])

class Other(WorldObj):
    def __init__(self):
        super().__init__('block', 'purple')

    def can_overlap(self):
        return False

    def can_move(self):
        return True

    def render(self, r):
        self._set_color(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])

class BlockGoal(WorldObj):
    def __init__(self):
        super().__init__('blockgoal', 'green')

    def can_overlap(self):
        return True

    def visible(self):
        return False 

    def render(self, r):
        self._set_color(r)
        r.drawPolygon([
            (0          , CELL_PIXELS/2),
            (CELL_PIXELS/2, CELL_PIXELS/2),
            (CELL_PIXELS/2,           0),
            (0          ,           0)
        ])

class MiniBlocksEnv(MiniGridEnv):
    """
    2D block world environment.

    Differences from grid world are:
    * Fully observable
    * Actions are in cardinal directions
    * Hide block goal location
    * Add agent tokens to observation
    * Add block objects that yield reward when reach block goal location
    * Blocks can be pushed around by self

    * Blocks can be pushed around by others
    """

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        right = 0
        down = 1
        left = 2
        up = 3
        none = 4

    def __init__(
        self,
        grid_size=16,
        max_steps=100,
        see_through_walls=True,
        seed=1337
    ):

        # Action enumeration for this environment
        self.actions = MiniBlocksEnv.Actions
        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))
        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        obs_size = (grid_size, grid_size, 3)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=obs_size,
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })
        # Range of possible rewards
        self.reward_range = (0, 1)
        # Renderer object used to render the whole grid (full-scale)
        self.grid_render = None
        # Renderer used to render observations (small-scale agent view)
        self.obs_render = None
        # Environment configuration
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls
        # Starting position and direction for the agent
        self.start_pos = None
        self.start_dir = None
        # Initialize the RNG
        self.seed(seed=seed)
        # Initialize the state
        self.reset()

    @property
    def front2_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agent_pos + self.dir_vec + self.dir_vec

    def gen_grid(self):
        if not self.see_through_walls:
            vis_mask = self.grid.process_vis(agent_pos=(AGENT_VIEW_SIZE // 2 , AGENT_VIEW_SIZE - 1))
        else:
            vis_mask = np.ones(shape=(self.grid.width, self.grid.height), dtype=np.bool)
        return self.grid, vis_mask

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        grid, vis_mask = self.gen_grid()
        # Encode the partially observable view into a numpy array
        image = grid.encode(agent_pos = self.agent_pos)
        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        obs = {
            'image': image,
        }
        return obs

    def _blockreward(self):
        return 1 - 0.9 * (self.step_count / self.max_steps)

    def step(self, action):
        self.step_count += 1
        reward = 0
        done = False

        #Movements are in cardinal directions
        if action in range(4):
            #Rotate based on direction chosen
            self.agent_dir = action
            # Get the position in front of the agent
            fwd_pos = self.front_pos
            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos

            # Agent-block interactions
            if fwd_cell != None and fwd_cell.type == 'block':
                # Get the position 2 in front of the agent
                fwd2_pos = self.front2_pos
                # Get the contents of the cell 2 in front of the agent
                fwd2_cell = self.grid.get(*fwd2_pos)
                # Move the block forward if can do so
                if fwd2_cell == None:
                    self.grid.set(*fwd2_pos, fwd_cell)
                    self.grid.set(*fwd_pos, None)
                    # Move the agent forward
                    self.agent_pos = fwd_pos
                #Check if moved block to a block goal location
                if fwd2_cell != None and fwd2_cell.type == 'blockgoal':
                    self.grid.set(*fwd2_pos, fwd_cell)
                    self.grid.set(*fwd_pos, None)
                    # Move the agent forward
                    self.agent_pos = fwd_pos
                    done = True
                    reward = self._blockreward()

            #Agent-goal interactions
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()

        elif action != self.actions.none:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        The agent is represented by `⏩`. A grid pixel is represented by 2-character
        string, the first one for the object and the second one for the color.
        """

        from copy import deepcopy

        def rotate_left(array):
            new_array = deepcopy(array)
            for i in range(len(array)):
                for j in range(len(array[0])):
                    new_array[j][len(array[0])-1-i] = array[i][j]
            return new_array

        def vertically_symmetrize(array):
            new_array = deepcopy(array)
            for i in range(len(array)):
                for j in range(len(array[0])):
                    new_array[i][len(array[0])-1-j] = array[i][j]
            return new_array

        # Map of object id to short string
        OBJECT_IDX_TO_IDS = {
            0: ' ',
            1: 'W',
            2: 'D',
            3: 'L',
            4: 'K',
            5: 'B',
            6: 'X',
            7: 'G',
            9: 'O',
            10: 'K'
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        # Map of color id to short string
        COLOR_IDX_TO_IDS = {
            0: 'R',
            1: 'G',
            2: 'B',
            3: 'P',
            4: 'Y',
            5: 'E'
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_IDS = {
            0: '⏩ ',
            1: '⏬ ',
            2: '⏪ ',
            3: '⏫ '
        }

        array = self.grid.encode(render_invisible = True)

        array = rotate_left(array)
        array = vertically_symmetrize(array)

        new_array = []

        for line in array:
            new_line = []

            for pixel in line:
                # If the door is opened
                if pixel[0] in [2, 3] and pixel[2] == 1:
                    object_ids = OPENDED_DOOR_IDS
                else:
                    object_ids = OBJECT_IDX_TO_IDS[pixel[0]]

                # If no object
                if pixel[0] == 0:
                    color_ids = ' '
                else:
                    color_ids = COLOR_IDX_TO_IDS[pixel[1]]

                new_line.append(object_ids + color_ids)

            new_array.append(new_line)

        # Add the agent
        new_array[self.agent_pos[1]][self.agent_pos[0]] = AGENT_DIR_TO_IDS[self.agent_dir]

        return "\n".join([" ".join(line) for line in new_array])