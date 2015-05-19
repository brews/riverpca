#   Copyright 2012-2014 S. Brewster Malevich <malevich@email.arizona.edu>
#
#   This is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>

"""
Classes to create and run a wallace problem.
"""

import numpy as np

def distance_sphere(lat1, long1, lat2, long2):
    """Get the distance between two lat/lon points (m).

    Modified from John Cook's beautiful code (http://www.johndcook.com/).
    """
    if lat1 == lat2 and long1 == long2:
        return 0
    earth_radius = 6378135
    degrees_to_radians = np.pi/180
    phi1 = (90 - lat1) * degrees_to_radians
    phi2 = (90 - lat2) * degrees_to_radians
    theta1 = long1 * degrees_to_radians
    theta2 = long2 * degrees_to_radians
    cos = (np.sin(phi1) * np.sin(phi2) * np.cos(theta1 - theta2) + 
           np.cos(phi1) * np.cos(phi2))
    arc = np.arccos( cos )
    return arc * earth_radius

class Problem(object):
    """Main control class for a wallace problem.

    Attributes:
        data: A Cube instance that the Problem will search over.
        agent_corral: The Corral instance or collection of agents that will be
            run over self.data to find a solution.
    """

    def __init__(self):
        """
        Args:
        
        Raises:
        """
        self.data = None
        self.agent_corral = Corral()

    def set_cube(self, *args, **kwargs):
        """ Sets a Cube instance for the Problem to work over.

        Args:
            *args: Passed to Cube constructor.
            **kwargs: Passed to Cube constructor.

        Returns:
            Nothing is returned.

        Raises:
        """
        self.data = Cube(*args, **kwargs)

    def add_agent(self, latlon, direction):
        """ Add a single agent to the problem's Corral.
        Args:
            latlon: A tuple with floats, (lat, lon) giving the latitude and 
                longitude from which the agent will begin its search.
            direction: Either "up" or "down" giving the direction the agent 
                will search the gadient.

        Returns:
            Nothing is returned.

        Raises:
        """
        yx = self.data.latlon2yx(latlon)
        self.agent_corral.append(Agent(self.data, yx, direction))

    def add_agentgrid(self, latlon, dlatdlon, nlatnlon, direction):
        """
        Args:
            latlon: A tuple, (lat, lon) giving the latitude and longitude from 
                which the agent will begin its search.
            dlatdlon: Tuple, (dlat, dlon), giving the width (dlon) of the grid 
                in longitude and height (dlat) of the grid in latitude.
            nlatnlon: Tuple, (nlat, nlon), giving the number of steps to have 
                between lat and dlat (nlat) and the number of steps between lon 
                and dlon (nlon) when placing agents.
            direction: The direction the agents will move in the grid. Either 
                "up" or "down".

        Returns:

        Raises:
        Maybe combine with self.add_agent?
        """
        yx = self.data.latlon2yx(latlon)
        yx2 = self.data.latlon2yx((latlon[0] + dlatdlon[0], latlon[1] + dlatdlon[1]))
        dydx = (yx[0] - yx2[0], yx[1] - yx2[1])
        self.agent_corral.append(AgentGrid(self.data, yx, dydx, nynx, direction))

    def clear_agents(self):
        """Remove all of the Agents that have been given to the Problem.
        """
        self.agent_corral.clear()

    def get_solution(self):
        """Return a Solution for a run agents.

        Returns:
            A list containing the solutions for each of the Agents or 
            AgentGrids in the problem.

        Raises:
        """
        return self.agent_corral.solution

    def run(self):
        """Have all agents in the corral search for a solution.

        Returns:
            Nothing is returned.
        """
        self.agent_corral.run()

    def get_positions(self, time=0, step=0):
        """ Get (lat, lon) positions for the Problem's agents.

        Args:
            time: The time slice from which to get the position. Default 
                is '0' or the first time slice.
            step: The step or iteration of the agent as it converges on a 
                olution. The default is iteration '0', or the starting 
                position. For the last position use '-1'. Keep in mind that 
                you can step backwards. For example, '-3', will give you the 
                position when the agent is 3 steps away from a solution.

        Returns:
            A list of location tuples, (lat, lon), giving the agent's 
            lat/lon-coordinate position.

        Raises:
            TypeError
        """
        # This is sacrilegious. Correct it someday.
        raw = self.agent_corral.get_position(time, step)
        flattened = []
        for i in raw:
            if type(i) == tuple:
                flattened.append(i)
            elif type(i) == list:
                [flattened.append(y) for y in i]
            else:
                raise TypeError
        return [(self.data.yx2latlon((i[0], i[1]))) for i in flattened]


class Corral(list):
    """A collection of Agent and AgentGrid objects to run as a search.

    Attributes:

    Also consider an ordered dict or dict subclass instead of list.
    We need some way to get solutions out of this, possibly by returning 
    Solution objects.
    """

    def __init__(self):
        """
        Args:

        Raises:
        """
        super(Corral, self).__init__()

    def clear(self):
        """ Clear out the corral.

        Raises:
        """
        self[:] = []

    def run(self):
        """ run() each item in the corral.

        Returns:
            Nothing is returned.

        Raises:
        """
        for i in self:
            i.run()
        self.solution = [i.solution for i in self]

    def get_position(self, time, step = 0):
        """Get the position of every Agent in the Corral.

        Args:
            time: The time slice from which to get the position.
            step: The step or iteration of the agent as it converges on a 
                olution. The default is iteration '0', or the starting 
                position. For the last position use '-1'. Keep in mind that 
                you can step backwards. For example, '-3', will give you the 
                position when the agent is 3 steps away from a solution.

        Returns:
            A list with location tuples, (y, x), giving the agents' 
            grid-coordinate position.

        Raises:
        """
        return [i.get_position(time, step) for i in self]


class Agent(object):
    """An individual agent object, which searches for a solution.

    Attributes:
        data: The Cube object that the agent will act on.
        start_location: A tuple, (y, x), with two numbers, y, the y or 
            latitudinal index for the position. Again, the index is 
            starting from 0. x, the x or longitudinal index for the 
            position, starting the index from 0.
        current_position: A tuple, (t, y, x), with three numbers, t, the index 
            for the position in time. Index starts from 0. y, the y or 
            latitudinal index for the position. Again, the index is 
            starting from 0. x, the x or longitudinal index for the 
            position, starting the index from 0.
        direction: The direction the agent should move along the gradient. 
            Either "up" or "down".
        solution: Solution for the agent.
    """

    def __init__(self, data, location, direction):
        """
        Args:
            data: The Cube object that the agent will act on.
            location: The starting location for the search. A tuple, (y, x), 
                with two numbers, y, the y or  latitudinal index for the 
                position. Again, the index is starting from 0. x, the x or 
                longitudinal index for the position, starting the index from 0.
            direction: The direction the agent should move along the gradient. 
                Either "up" or "down".

        Raises:
        """
        self.solution = None
        self.data = data
        self.start_location = location
        self.current_position = (0,) + location
        self.direction = direction

    def run(self):
        """Have the agent search for a solution for every time-slice.

        Returns:
            A list. The first dimension is for each time step in the agent's 
            data. The second dimension gives a list with the path the agent 
            used to each its solution. Each of the items in this list are a 
            tuple ((t, y, x), value).

        Raises:
        """
        time_range = range(self.data.get_no_time_obs())
        self.solution = [[i for i in self.greedy_search( (t,) + self.start_location)] for t in time_range]

    def get_best_adjacent(self, position):
        """Get the 'best' position & value of those adjacent to position.

        Args:
            position: A tuple, (t, y, x), with three numbers, t, the index for 
                the position in time. Index starts from 0. y, the y or 
                latitudinal index for the position. Again, the index is 
                starting from 0. x, the x or longitudinal index for the 
                position, starting the index from 0.

        Returns:
            A (t, y, x) position tuple.

        Raises:
        """
        neighbors = self.data.get_adjacent(position)
        best = (position, self.data.get_value(position))
        for i in neighbors:
            if i[1] is np.nan:
                break
            else:
                # Before we hit this we need error check that self.direction is
                # either "down" or "up".
                if (self.direction == "down") and (i[1] < best[1]):
                    best = i
                elif (self.direction == "up") and (i[1] > best[1]):
                    best = i
                else:
                    continue
        return(best[0])

    def greedy_search(self, position=None):
        """Agent moves with greedy search in a single direction.

        Args:
            position: A tuple, (t, y, x), with three numbers, t, the index for 
                the position in time. Index starts from 0. y, the y or 
                latitudinal index for the position. Again, the index is 
                starting from 0. x, the x or longitudinal index for the 
                position, starting the index from 0. Default makes position 
                the agent's current position.

        Returns:
            A generator of giving ((t, y, x), value) for each step in the 
            search until a solution is reached.

        Raises:
        """
        if position is None:
            position = self.current_position
        previous_position = None
        while previous_position != position:
            next_move = self.get_best_adjacent(position)
            previous_position = position
            position = next_move
            yield position, self.data.get_value(position)

    def get_position(self, time, step = 0):
        """Get the position of the Agent.

        Args:
            time: The self.data time slice from which to get the position.
            step: The step or iteration of the agent as it converges on a 
                olution. The default is iteration '0', or the starting 
                position. For the last position use '-1'. Keep in mind that 
                you can step backwards. For example, '-3', will give you the 
                position when the agent is 3 steps away from a solution.

        Returns:
            A location tuple, (y, x), giving the agent's grid-coordinate 
            position.

        Raises:
        """
        return self.solution[time][step][0][1:]


class AgentGrid(list):
    """Grid collection of agents, which together search for a solution.

    Attributes:
        data: A Cube instance which the search will be performed on.
        yx: Tuple, (yx), giving upper left and right index for the grid.
        dydx: Tuple, (dy, dx), giving the index height and width of the grid.
        nynx: Tuple, (ny, nx), giving the number of steps to place agents 
            between y/x and dy/dx.
        direction: The direction the agents will move in the grid.
        solution: Solution for the agent.
    """

    def __init__(self, data, yx, dydx, nynx, direction):
        """
        Args:
            data: A cube object which the search will be performed on.
            yx: Tuple, (y, x), giving the upper left cordinate x-index for the 
                grid and the upper left cordinate y-index for the grid.
            dydx: Tuple, (dy, dx), giving the width (dx) of the grid in 
                index-units and height (dy) of the grid in index-units
            nynx: Tuple, (ny, nx), giving the number of steps to have between x 
                and dx (nx) and the number of steps between y and dy (ny) when 
                placing agents.
            direction: The direction the agents will move in the grid. Either 
                "up" or "down".

        Raises:
        """
        super(AgentGrid, self).__init__()
        self.solution = None
        self.data = data
        self.direction = direction
        self.yx = yx; self.dydx = dydx; self.nynx = nynx
        width_range = np.arange(self.yx[1], self.yx[1] + self.dydx[1] + 1, self.nynx[1])
        height_range = np.arange(self.yx[0], self.yx[0] + self.dydx[0] + 1, self.nynx[0])
        for i in height_range:
            for j in width_range:
                self.append(Agent(self.data, (i, j), self.direction))

    def run(self):
        """Have the agent search for a solution for every time-slice.
        """
        for i in self:
            i.run()
        self.solution = [i.solution for i in self]

    def get_position(self, time, step = 0):
        """Get the position of every Agent in the grid.

        Args:
            time: The time slice from which to get the position.
            step: The step or iteration of the agent as it converges on a 
                olution. The default is iteration '0', or the starting 
                position. For the last position use '-1'. Keep in mind that 
                you can step backwards. For example, '-3', will give you the 
                position when the agent is 3 steps away from a solution.

        Returns:
            A list with yx tuples, (y, x), giving the agents' 
            grid-coordinate position.

        Raises:
        """
        return [i.get_position(time, step) for i in self]


class Cube(object):
    """A three-dimensional collection of surfaces for a given problem.

    Attributes:
        data: A 3-D Numpy array containing the data of interest.
        _index_order: A dictionary giving the order of latitude, longitude and 
            time indexes in self.data.
        _latitude: A 2-D Numpy array with the same shape as self.data's position
            dimensions, giving the latitude for each point in self.data.
        _longitude: A 2-D Numpy array with the same shape as self.data's 
            position dimensions, giving the longitude for each point in self.data.
    """

    def __init__(self, data, lat, lon, lat_index, lon_index, time_index):
        """
        Args:
            data: A 3-D Numpy array containing the data of interest.
            lat_index: Integer giving the index or dimension which marks the 
                latitudinal coordinate in `data`.
            lon_index: Integer giving the index or dimension which marks the 
                longitudinal coordinate in `data`.
            time_index: Integer giving the index or dimension which marks the 
                progression of time in `data`.
            lat: A 2-D Numpy array with the same shape as self.data's position 
                dimensions, giving the latitude for each point in self.data.
            lon: A 2-D Numpy array with the same shape as self.data's position 
                dimensions, giving the longitude for each point in self.data.

        Raises:
        """
        # TODO(sbm): Need to error-check the conditions of our arguements.
        self._index_order = {"lat": lat_index, 
                             "lon": lon_index, 
                             "time": time_index}
        self._latitude = lat
        self._longitude = lon
        self.data = data


    def _get_index(self, x):
        """ Put index values in the correct order to read from self.data.

        Args:
            x - A tuple tuple giving the (y, x) index we desire. An optional 
                third element giving the time index is possible: (t, y, x).

        Returns:
            A tuple give the y, x, and optional time indexes in the correct 
            order for self.data.

        Raises:
        """
        l = None
        if len(x) == 2:
            if self._index_order["lat"] > self._index_order["lon"]:
                l = [x[1], x[0]]
            else:
                l = [x[0], x[1]]
        elif len(x) == 3:
            ingoing = {"lat": x[1], "lon": x[2], "time": x[0]}
            l = [None, None, None]
            for k in self._index_order.keys():
                l[self._index_order[k]] = ingoing[k]            
        return tuple(l)


    def get_value(self, position):
        """Get the values for point at (t, y, x).

        Args:
            position: A tuple, (t, y, x), with three integers: t, the index for 
                the position in time. Index starts from 0. y, the y or 
                latitudinal index for the position. Again, the index is 
                starting from 0. x, the x or longitudinal index for the 
                position, starting the index from 0.

        Returns:
            A simple numerical value of the given position in time and space.

        Raises:
        """
        z = np.nan
        try:
            z = self.data[self._get_index(position)]
        except IndexError:  # Peeks over the edge of the grid-data.
            pass
        return(z)


    def get_adjacent(self, position):
        """ Get the grid cells that are adjacent to a position.

        Args:
            position: A tuple, (t, y, x), with three numbers, t, the index for 
                the position in time. Index starts from 0. y, the y or 
                latitudinal index for the position. Again, the index is 
                starting from 0. x, the x or longitudinal index for the 
                position, starting the index from 0.

        Returns:
            A list of tuples where each tuple, ((t, y, x), value), has the 
            x-index, y-index, and grid value of valid adjacent (given 
            Queen-like movement) positions.

        Raises:
        """
        current_value = self.get_value(position)
        # Adjacent is defined by queen-like movement.
        potential_neighbors = [(position[0], position[1] + 1, position[2] - 1),
                               (position[0], position[1] + 1, position[2]),
                               (position[0], position[1] + 1, position[2] + 1),
                               (position[0], position[1], position[2] - 1),
                               (position[0], position[1], position[2] + 1),
                               (position[0], position[1] - 1, position[2] - 1),
                               (position[0], position[1] - 1, position[2]),
                               (position[0], position[1] - 1, position[2] + 1)]                               
        neighbors = []
        for i in potential_neighbors:
            v = self.get_value(i)
            if v is np.nan:
                break
            else:
                neighbors.append( (i, v) )
        return(neighbors)


    def get_no_time_obs(self):
        """Get the number of observations through time (the first dimension).
        """
        return self.data.shape[self._index_order["time"]]


    def yx2latlon(self, yx):
        """ Convert (y-index, x-index) grid position to (latitude, longitude).

        Args:
            yx: A tuple giving (y, x)-index for a position in the cube.

        Returns:
            A tuple with (latitude, longitude) for the given an xy index tuple.

        Raises:
        """
        i = self._get_index(yx)
        return (self._latitude[i], self._longitude[i])


    def latlon2yx(self, latlon):
        """ Convert (latitude, longitude) to nearest cube grid index position.

        Args:
            latlon: A tuple giving (latitude, longitude) for a position in the 
                cube.

        Returns:
            A tuple with the yx index nearest to the given (latitude, longitude)
                tuple and the distance to the point in meters.

        Raises:
        """
        outgoing = [(self._latitude.flat[i], self._longitude.flat[i]) for i in range(self._latitude.size)]
        dists = np.array([distance_sphere(latlon[0], latlon[1], x[0], x[1]) for x in outgoing])
        dists.shape = self._latitude.shape
        #TODO(sbm): What is we have more than one match?
        out = np.where(dists == dists.min())
        return tuple(int(i) for i in out)


class ProblemError(Exception):
    """Raised when there are issues with wallace Problem objects.

    Attributes:
        expression: Input expression in which the error occurred.
        message: Explanation of the error.
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message