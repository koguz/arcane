import random
from statistics import mean
from matplotlib import pyplot as plt


class Grid:

    def __init__(self, two_power_size, roughness, placing_threshold: (int or float) = None, placing_choice: int = None):
        self.grid = [[]]  # list(list[int])
        self.grid_width = (2 ** two_power_size) + 1
        self.max = self.grid_width - 1
        self.roughness = roughness
        self.fill_empty_grid()
        self.grid_pos_list = list()
        self.create_grid_pos_list()
        self.divide_grid(self.max)
        # ! Open them if you want to see avg and max values, and the grid itself (debug purposes)
        # print("Max value in grid is: ", max([y for x in self.grid for y in x]))
        # print("Avg value in grid is: ", mean([y for x in self.grid for y in x]))
        # self.print_grid()
        # ? For plotting histogram of grid data
        # self.plot_histogram()
        self.placeable_node_size = 0
        self.place_threshold = 0
        self.placeable_grid_positions = list()
        # ? Placing the nodes to the grid according to a selection (using placing_threshold OR using grid values)
        # """
        if placing_threshold is None:
            if placing_choice is None:
                print("Do you want to place the nodes according to a threshold?")
                placing_selection = int(input("Enter 0 for YES.\n"
                                              "Enter 1 for NO. (They will be scattered according to the grid "
                                              "values.): \n"))
                if placing_selection == 0:
                    input_threshold = float(
                        input("Please give a threshold value for placing the nodes into the grid (int or float): "))
                    self.place_objects(input_threshold)
                    self.place_threshold = input_threshold
                else:
                    input_threshold = float(
                        input("Please give a threshold value nonetheless to save it for a future use of placing: "))
                    self.place_threshold = input_threshold
                    self.place_objects_wrt_grid_values()
            elif placing_choice == 1:
                self.place_threshold = 0
                self.place_objects_wrt_grid_values()
        else:
            self.place_objects(placing_threshold)
            self.place_threshold = placing_threshold
        # """

    def print_grid(self):
        print("- - - - - - - - -")
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                print(self.grid[i][j], end=' ')
            print()
        print("- - - - - - - - -")

    # * Sets x,y position in self.grid
    def set_xy_value(self, x: int, y: int, value):
        if value < 0:
            # TODO: change to 0
            self.grid[x][y] = value
        else:
            self.grid[x][y] = value  # TODO: integer değerler istersek "int(value)" yaparız.

    # * Gets x,y position in self.grid
    def get_xy_value(self, x: int, y: int) -> int:
        if x < 0 or x > self.max or y < 0 or y > self.max:
            return -1
        return self.grid[x][y]

    def fill_empty_grid(self):
        for x in range(self.grid_width):
            self.grid[x] = [num for num in range(self.grid_width)]
            for y in range(self.grid_width):
                self.grid[x][y] = 0
            if x != self.grid_width-1:
                self.grid.append([])

        # For constant seed values
        """
        self.set_xy_value(0, 0, 4)
        self.set_xy_value(self.max, 0, 4)  # self.max / 2)
        self.set_xy_value(self.max, self.max, 4)  # 0)
        self.set_xy_value(0, self.max, 4)
        """

        # For int normally distributed seed values
        """
        self.set_xy_value(0, 0, int(random.normalvariate(self.max*2, 1.0)))
        self.set_xy_value(self.max, 0, int(random.normalvariate(self.max*2, 1.0)))  # self.max / 2)
        self.set_xy_value(self.max, self.max, int(random.normalvariate(self.max*2, 1.0)))  # 0)
        self.set_xy_value(0, self.max, int(random.normalvariate(self.max*2, 1.0)))  # self.max / 2)
        """

        # For int uniform seed values

        self.set_xy_value(0, 0, int(random.uniform(0, self.max)))
        self.set_xy_value(self.max, 0, int(random.uniform(0, self.max)))  # self.max / 2)
        self.set_xy_value(self.max, self.max, int(random.uniform(0, self.max)))  # 0)
        self.set_xy_value(0, self.max, int(random.uniform(0, self.max)))  # self.max / 2)

        # For float uniform seed values
        """
        self.set_xy_value(0, 0, random.uniform(0, self.max))
        self.set_xy_value(self.max, 0, random.uniform(0, self.max))  # self.max / 2)
        self.set_xy_value(self.max, self.max, random.uniform(0, self.max))  # 0)
        self.set_xy_value(0, self.max, random.uniform(0, self.max))  # self.max / 2)
        """

    def divide_grid(self, size):
        half = int(size / 2)
        scale = self.roughness * size

        if half < 1:
            return

        # Square
        for y in range(half, self.max, int(size)):
            for x in range(half, self.max, int(size)):
                # For Normal distribution
                # s_scale = random.normalvariate(0.0, 1.0) * scale * 2 - scale
                # For Uniform distribution
                s_scale = random.uniform(0, 1) * scale * 2 - scale
                self.square_step(x, y, half, s_scale)

        # Diamond
        for y in range(0, self.max + 1, half):
            for x in range(int((y + half) % size), self.max + 1, int(size)):
                # For Normal distribution
                # d_scale = random.normalvariate(0.0, 1.0) * scale * 2 - scale
                # For Uniform distribution
                d_scale = random.uniform(0, 1) * scale * 2 - scale
                self.diamond_step(x, y, half, d_scale)

        self.divide_grid(size / 2)

    def square_step(self, x, y, size, scale):
        top_left = self.get_xy_value(x - size, y - size)  # self.grid[x - size][y - size]
        top_right = self.get_xy_value(x + size, y - size)
        bottom_left = self.get_xy_value(x + size, y + size)
        bottom_right = self.get_xy_value(x - size, y + size)

        average = ((top_left + top_right + bottom_left + bottom_right) / 4)
        # TODO: change this to "int" if you want to get int values
        self.set_xy_value(x, y, int(average + scale))
        # self.grid[x][y] = average + scale

    def diamond_step(self, x, y, size, scale):
        top = self.get_xy_value(x, y - size)
        right = self.get_xy_value(x + size, y)
        bottom = self.get_xy_value(x, y + size)
        left = self.get_xy_value(x - size, y)

        average = ((top + right + bottom + left) / 4)
        # TODO: change this to "int" if you want to get int values
        self.set_xy_value(x, y, int(average + scale))

    def create_grid_pos_list(self):
        for x in range(self.grid_width):
            for y in range(self.grid_width):
                self.grid_pos_list.append((x, y))

    # ? Using the placing_threshold to distribute the nodes into their position if the value is bigger
    def place_objects(self, _placing_threshold):
        for x in range(self.grid_width):
            for y in range(self.grid_width):
                if self.grid[x][y] > _placing_threshold:
                    self.placeable_node_size += 1
                    # ? USE BELOW for original grid placing without a little random addition to positions
                    # self.placeable_grid_positions.append((x, y))
                    # ! COMMENT BELOW if you need original grid representation
                    random_x_add = random.uniform(-0.25, 0.25)
                    random_y_add = random.uniform(-0.25, 0.25)
                    self.placeable_grid_positions.append((x + random_x_add, y + random_y_add))

    # ? Using the grid values to randomly distribute the nodes with the center of the (x, y) position
    def place_objects_wrt_grid_values(self):
        for x in range(self.grid_width):
            for y in range(self.grid_width):

                if self.grid[x][y] <= 0:
                    self.placeable_node_size += 0
                    cell_size = 0
                else:
                    self.placeable_node_size += self.grid[x][y]
                    cell_size = self.grid[x][y]

                random_pos_list = []
                for index in range(int(cell_size)):
                    random_x_addition = random.uniform(-0.95, 0.95)
                    random_y_addition = random.uniform(-0.95, 0.95)
                    random_pos_tuple = (x + random_x_addition, y + random_y_addition)
                    if random_pos_tuple not in random_pos_list:
                        random_pos_list.append(random_pos_tuple)
                        self.placeable_grid_positions.append(random_pos_tuple)
                    else:
                        # TODO: position'ları belli bir threshold (node drawing size) ile karşılaştırıp
                        #  ona göre kontrol eden bool method
                        while random_pos_tuple in random_pos_list:
                            random_x_addition = random.uniform(-0.95, 0.95)
                            random_y_addition = random.uniform(-0.95, 0.95)
                            random_pos_tuple = (x + random_x_addition, y + random_y_addition)
                            if random_pos_tuple not in random_pos_list:
                                random_pos_list.append(random_pos_tuple)
                                self.placeable_grid_positions.append(random_pos_tuple)
                                break
                # self.placeable_grid_positions.extend(random_pos_list)

    def plot_histogram(self):
        fig, ax = plt.subplots()
        plt.hist(list(self.grid))
        # ax.hist(list(self.grid), bins=max([y for x in self.grid for y in x]), linewidth=0.5, edgecolor="white")
        plt.show()


# ! Test Method
# a = Grid(3, 0.4)
# a.print_grid()
# print(a.placeable_node_size)
