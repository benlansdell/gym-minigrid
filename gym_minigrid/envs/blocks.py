from gym_minigrid.miniblocks import *
from gym_minigrid.register import register
import numpy.random as rand

#####################################
# Basic maze environment generation #
#####################################

class BlockMazeEnv(MiniBlocksEnv):
    """
    Basic maze blocks environment: three rooms of varying size and relation to one another
    Block goal in one room, agent starts in another room
    """

    def __init__(self, size=16):
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Note: generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()

        #Pick dimensions of the rooms. Somewhere between 5 and 7
        d1 = 5+rand.randint(2)
        d2 = 5+rand.randint(2)
        offset = rand.randint(2)
        rotate = rand.randint(4)
        #Choose goal location in room 2
        bx, by = 2+rand.randint(d1-4), 2+rand.randint(d1-4)
        gx, gy = d1+rand.randint(d2-2), offset+1+rand.randint(d2-2)
        # Create an empty grids
        d3 = d1+d2
        #print(d1-1, offset, d1+d2-1, offset+d2, 'width', width, 'height', height)
        self.grid = Grid(self.grid_size, self.grid_size)
        self.rotate = rotate
        self.viable_width = d3
        self.viable_height = d3
        #self.grid = Grid(width, height)
        #Make rooms
        self.grid.wall_rect(0, 0, self.grid_size, self.grid_size)
        self.grid.wall_rect(0, 0, self.viable_width, self.viable_height)
        self.grid.wall_rect(0, 0, d1, d1)
        self.grid.wall_rect(d1-1, offset, d2+1, d2)
        #Make doors between the rooms
        self.grid.set(2,d1-1,Door('red', True))
        self.grid.set(d1+1,offset+d2-1,Door('red', True))
        self.grid.set(d1-1,min(offset+2, d1-2),Door('red', True))
        #Place the goals and blocks
        self.grid.set(gx, gy, VisibleBlockGoal())
        self.grid.set(bx, by, Block())
        #Rotate by some amount
        for i in range(rotate):
            self.grid = self.grid.rotate_left()
        # Place the agent in the top-left corner
        self.start_pos = (1, 1)
        self.start_dir = 0
        self.mission = "push block to goal square"

###########################
## Register environments ##
###########################

#Basic maze
register(
    id='MiniGrid-BlockMaze-v0',
    entry_point='gym_minigrid.envs:BlockMazeEnv'
)

##########################
## Phase 3 environments ##
##########################

#class BlocksEnv(MiniBlocksEnv):
#    """
#    Empty blocks environment, no obstacles, sparse reward
#    """
#
#    def __init__(self, size=8):
#        super().__init__(
#            grid_size=size,
#            max_steps=4*size*size,
#            # Set this to True for maximum speed
#            see_through_walls=True
#        )
#
#    def _gen_grid(self, width, height):
#        # Create an empty grid
#        self.grid = Grid(width, height)
#
#        # Generate the surrounding walls
#        self.grid.wall_rect(0, 0, width, height)
#
#        # Place the agent in the top-left corner
#        self.start_pos = (1, 1)
#        self.start_dir = 0
#
#        # Place a goal square in the bottom-right corner
#        self.grid.set(width - 2, height - 2, BlockGoal())
#
#        #Add a block
#        self.grid.set(width - 4, height - 4, Block())
#
#        self.mission = "push block to goal square"
#
#class BlocksEnv6x6(BlocksEnv):
#    def __init__(self):
#        super().__init__(size=6)
#
#class BlocksEnv16x16(BlocksEnv):
#    def __init__(self):
#        super().__init__(size=16)


##########################
## Phase 1 environments ##
##########################

#class BlocksFamEnv(BlocksEnv):
#    """
#    Empty blocks environment, no obstacles, no reward
#    """
#
#    def _gen_grid(self, width, height):
#        # Create an empty grid
#        self.grid = Grid(width, height)
#
#        # Generate the surrounding walls
#        self.grid.wall_rect(0, 0, width, height)
#
#        # Place the agent in the top-left corner
#        self.start_pos = (1, 1)
#        self.start_dir = 0
#
#        # Place a goal square in the bottom-right corner
#        self.grid.set(width - 2, height - 2, BlockGoal())
#
#        #Add a block
#        self.grid.set(width - 4, height - 4, Block())
#
#        self.mission = "push block to goal square"
#
#class BlocksFamEnv6x6(BlocksFamEnv):
#    def __init__(self):
#        super().__init__(size=6)
#
#class BlocksFamEnv16x16(BlocksFamEnv):
#    def __init__(self):
#        super().__init__(size=16)
#
###########################
### Phase 2 environments ##
###########################
#
###########################
### Watching other agent ##
###########################
#
#class BlocksOtherEnv(MiniBlocksOtherEnv):
#    """
#    Empty blocks environment, no obstacles, sparse reward
#    """
#
#    def __init__(self, size=8):
#        super().__init__(
#            grid_size=size,
#            max_steps=4*size*size,
#            # Set this to True for maximum speed
#            see_through_walls=True
#        )
#
#    def _gen_grid(self, width, height):
#        # Create an empty grid
#        self.grid = Grid(width, height)
#
#        # Generate the surrounding walls
#        self.grid.wall_rect(0, 0, width, height)
#
#        # Place the agent in the top-left corner
#        self.start_pos = (1, 1)
#        self.start_dir = 0
#
#        # Place a goal square in the bottom-right corner
#        self.grid.set(width - 2, height - 2, BlockGoal())
#
#        #Add a block
#        self.grid.set(width - 4, height - 4, Block())
#
#        self.mission = "push block to goal square"
#
#class BlocksOtherEnv6x6(BlocksOtherEnv):
#    def __init__(self):
#        super().__init__(size=6)
#
#class BlocksOtherEnv16x16(BlocksOtherEnv):
#    def __init__(self):
#        super().__init__(size=16)
#
################################################
### Watching no other agent (ghost condition) ##
################################################
#
#class BlocksGhostEnv(MiniBlocksGhostEnv):
#    """
#    Empty blocks environment, no obstacles, sparse reward
#    """
#
#    def __init__(self, size=8):
#        super().__init__(
#            grid_size=size,
#            max_steps=4*size*size,
#            # Set this to True for maximum speed
#            see_through_walls=True
#        )
#
#    def _gen_grid(self, width, height):
#        # Create an empty grid
#        self.grid = Grid(width, height)
#
#        # Generate the surrounding walls
#        self.grid.wall_rect(0, 0, width, height)
#
#        # Place the agent in the top-left corner
#        self.start_pos = (1, 1)
#        self.start_dir = 0
#
#        # Place a goal square in the bottom-right corner
#        self.grid.set(width - 2, height - 2, BlockGoal())
#
#        #Add a block
#        self.grid.set(width - 4, height - 4, Block())
#
#        self.mission = "push block to goal square"
#
#class BlocksGhostEnv6x6(BlocksGhostEnv):
#    def __init__(self):
#        super().__init__(size=6)
#
#class BlocksGhostEnv16x16(BlocksGhostEnv):
#    def __init__(self):
#        super().__init__(size=16)

##Others
#
#register(
#    id='MiniGrid-Blocks-6x6-v0',
#    entry_point='gym_minigrid.envs:BlocksEnv6x6'
#)
#
#register(
#    id='MiniGrid-Blocks-8x8-v0',
#    entry_point='gym_minigrid.envs:BlocksEnv'
#)
#
#register(
#    id='MiniGrid-Blocks-16x16-v0',
#    entry_point='gym_minigrid.envs:BlocksEnv16x16'
#)
#
#########
#
#register(
#    id='MiniGrid-BlocksFam-6x6-v0',
#    entry_point='gym_minigrid.envs:BlocksFamEnv6x6'
#)
#
#register(
#    id='MiniGrid-BlocksFam-8x8-v0',
#    entry_point='gym_minigrid.envs:BlocksFamEnv'
#)
#
#register(
#    id='MiniGrid-BlocksFam-16x16-v0',
#    entry_point='gym_minigrid.envs:BlocksFamEnv16x16'
#)
#
#########
#
#register(
#    id='MiniGrid-BlocksOther-6x6-v0',
#    entry_point='gym_minigrid.envs:BlocksOtherEnv6x6'
#)
#
#register(
#    id='MiniGrid-BlocksOther-8x8-v0',
#    entry_point='gym_minigrid.envs:BlocksOtherEnv'
#)
#
#register(
#    id='MiniGrid-BlocksOther-16x16-v0',
#    entry_point='gym_minigrid.envs:BlocksOtherEnv16x16'
#)
#
#########
#
#register(
#    id='MiniGrid-BlocksGhost-6x6-v0',
#    entry_point='gym_minigrid.envs:BlocksGhostEnv6x6'
#)
#
#register(
#    id='MiniGrid-BlocksGhost-8x8-v0',
#    entry_point='gym_minigrid.envs:BlocksGhostEnv'
#)
#
#register(
#    id='MiniGrid-BlocksGhost-16x16-v0',
#    entry_point='gym_minigrid.envs:BlocksGhostEnv16x16'
#)
