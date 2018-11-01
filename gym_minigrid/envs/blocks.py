from gym_minigrid.miniblocks import *
from gym_minigrid.register import register

class BlocksEnv(MiniBlocksEnv):
    """
    Empty blocks environment, no obstacles, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (1, 1)
        self.start_dir = 0

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, BlockGoal())

        #Add a block
        self.grid.set(width - 4, height - 4, Block())

        self.mission = "push block to goal square"

class BlocksEnv6x6(BlocksEnv):
    def __init__(self):
        super().__init__(size=6)

class BlocksEnv16x16(BlocksEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-Blocks-6x6-v0',
    entry_point='gym_minigrid.envs:BlocksEnv6x6'
)

register(
    id='MiniGrid-Blocks-8x8-v0',
    entry_point='gym_minigrid.envs:BlocksEnv'
)

register(
    id='MiniGrid-Blocks-16x16-v0',
    entry_point='gym_minigrid.envs:BlocksEnv16x16'
)
