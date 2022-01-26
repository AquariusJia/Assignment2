import json
from clustering_numpy import *
from clustering import *
from math import sqrt
from operator import sub
import matplotlib.pyplot as plt
import requests
import os
from argparse import ArgumentParser
from collections import Counter


def load_tracksfile(filename):
    """Load_tracks function
    Parameters
    ----------
    filename: str
        name of the file that the function loads
    Returns
    ----------
    object
        calling the Tracks object for further analysis

    Examples
    ----------
    >>> tracks = query_tracks(start=(2, 3), end=(4, 2), min_steps_straight=1, max_steps_straight=6, n_tracks=30, \
                         save=True)
    >>> print(tracks)
    <Tracks: 17 from (2, 3) to (4, 2)>

    """

    # The load_tracksfile function takes only a Path to a JSON filename
    with open(filename, 'r') as target_file:
        tracks = Tracks(filename)

    # Verify the digits of the chaincode
    for k in range(len(tracks.a_track_list)):
        for i in tracks.a_track_list[k].cc:
            if int(i) > 5 or int(i) < 1:
                raise ValueError("The chaincode digits must be from 1 to 4")

    # Verify the rode-type code
    for k in range(len(tracks.a_track_list)):
        for i in tracks.a_track_list[k].road:
            if i not in ["r", "l", "m"]:
                raise ValueError("The rode-type code must be r, l, or m")

    # Verify the terrain-type code
    for k in range(len(tracks.a_track_list)):
        for i in tracks.a_track_list[k].terrain:
            if i not in ["d", "g", "p"]:
                raise ValueError("The rode-type code must be d, g, or p")

    return tracks


def query_tracks(start=[0, 0], end=[299, 299], min_steps_straight=1, max_steps_straight=6, n_tracks=300, save=bool):
    """ This function allows the user to download the generated webapp data as a json file, which is a dictionary.
    When the save option of the function is set to true: the obtained data is saved as a json file with the
    parameters of the function arguments rather than the default webapp parameters.
    Parameters
    ----------
    start: list
        the coordinate of the start point, defaults to [0, 0]
    end: list
        the coordinate of the end point, defaults to [299, 299]
    min_steps_straight: int
        the minimum steps needed to query, defaults to 1
    max_steps_straight: int
        the maximum steps needed to query, defaults to 6
    n_tracks: int
        the number of tracks needed to query, defaults to 300
    save: bool
        tell if the function will save the json file
    Returns
    ----------
    save the file in the local directory
    """

    """Validation"""
    for i in start:
        if i < 0:
            raise ValueError("The queried coordinates {} should not be negative".format(start))

    for i in end:
        if i < 0:
            raise ValueError("The queried coordinates {} should not be negative".format(end))

    if min_steps_straight < 1:
        raise ValueError("The minimum steps straight value cannot be less than 1")

    if max_steps_straight < min_steps_straight:
        raise ValueError("The maximum steps straight value cannot be less than the minimum steps straight value")

    if n_tracks < 1:
        raise ValueError("The number of tracks cannot be less than 1")

    if isinstance((start, end, min_steps_straight, max_steps_straight, n_tracks), int):
        raise TypeError("All input arguments apart from save must be integers")

    # String that is a json representation of webapp information
    response = requests.get(
        "https://ucl-rse-with-python.herokuapp.com/road-tracks/tracks/?start_point_x={}&start_point_y={}"
        "&end_point_x={}&end_point_y={}&min_steps_straight={}&max_steps_straight={}&n_tracks={}".format(
            start[0], start[1], end[0], end[1], min_steps_straight, max_steps_straight, n_tracks)
    )

    # Printing the response of the website into a file
    # with open('downloaded.json', 'w') as output_file:
    #     output_file.write(response.text)

    # Output the data as a dictionary
    # data = json.loads(response.text)
    data = response.json()

    # if 'cc' and 'elevation' and 'terrain' and 'road' not in data['tracks'][0]:
    #     raise KeyError("Wrong json schema loaded")

    # Getting the current datetime
    datetime = data['metadata']['datetime']

    # Delete the ":" and "-" in the date string
    datetime = datetime.replace(":", "")
    datetime = datetime.replace("-", "")

    # Re-writing downloaded json file with modifications according to function arguments
    # with open('rewritten.json', 'w') as new_output_files:
    #     json.dump(data, new_output_files)

    # If save argument is set to true, then an output json file with the track object will be saved
    if save == True:
        # Re-writing downloaded json file with modifications according to function arguments
        with open('rewritten.json', 'w') as new_output_files:
            json.dump(data, new_output_files)
        output_path = 'tracks' + "_" + str(datetime) + "_" + str(n_tracks) + "_" + str(start[0]) + "_" + str(start[1]) \
                      + "_" + str(end[0]) + "_" + str(end[1]) + '.json'

        # Renaming the file
        os.rename("rewritten.json", output_path)
        tracks = load_tracksfile(output_path)
        return tracks
    else:
        return response.json()

    # if 'cc' and 'elevation' and 'terrain' and 'road' not in data['tracks'][0]:
    #     raise KeyError("Wrong json schema loaded")

    # tracks = load_tracksfile(output_path)


def connected_to_internet(url="https://ucl-rse-with-python.herokuapp.com/road-tracks/tracks/", timeout=5):
    """
    Validating when user tries to query webapp without internet connection
    """

    try:
        _ = requests.head(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        print("No internet connection available.")
    return False


class Tracks:
    def __init__(self, track_filename):

        self.track_filename = track_filename

        with open(self.track_filename, 'r') as f:
            jdic = json.load(f)

        date = jdic['metadata']['datetime']
        end = tuple(jdic['metadata']['end'])
        start = tuple(jdic['metadata']['start'])
        map_size = jdic['metadata']['mapsize']
        tracks_dic = jdic['tracks']

        a_track_list = []
        for i in range(len(tracks_dic)):
            a_track = SingleTrack(start,
                                  tracks_dic[i]['cc'],
                                  tracks_dic[i]['elevation'],
                                  tracks_dic[i]['road'],
                                  tracks_dic[i]['terrain'])
            a_track_list.append(a_track)

        self.a_track_list = a_track_list
        self.tracks_length = len(tracks_dic)
        self.start = start
        self.end = end
        self.map_size = map_size
        self.date = date

    def __len__(self):
        """__len__ function which returns the number of Track objects.
        Parameters
        ----------
        self
        Returns
        -------
        tracks_length: int
            The number of SingleTrack object in the tacks collection
        Examples
        --------
        >>> len(Tracks('short_tracks.json'))
        5
        """
        return self.tracks_length

    def __str__(self):
        """__len__ function which returns the number of Track objects.
        Parameters
        ----------
        self
        Returns
        -------
        tracks_length: int
            The number of SingleTrack object in the tacks collection
        Examples
        --------
        >>> len(Tracks('short_tracks.json'))
        5
        """
        return "<Tracks: {} from {} to {}>".format(self.tracks_length, self.start, self.end)

    def greenest(self):
        """Find the track in the track collection with the least CO2 emission

        Returns
        ----------
        object
            the SingleTrack object in the tracks collections with least CO2 emission
        Examples
        ----------
        >>> Tracks('short_tracks.json').greenest().cc
        '21144'
        """

        # Store list with CO2 emission for all tracks
        a_track_co2_list = []
        for a_track in self.a_track_list:
            a_track_co2_list.append(a_track.co2())

        # Find minimal value in CO2 list
        min_co2 = min(a_track_co2_list)

        # Find index of minimal value and corresponding track
        min_co2_index = a_track_co2_list.index(min_co2)
        greenest_track = self.a_track_list[min_co2_index]

        return greenest_track

    def fastest(self):
        """
        Find the track in the track collection with the least CO2 emission
        Returns
        -------
        object
            the SingleTrack object in the tracks collections with least CO2 emission
        Examples
        --------
        >>> Tracks('short_tracks.json').greenest().cc
        '21144'
        """
        # Store list with time for all tracks
        a_track_time_list = []
        for a_track in self.a_track_list:
            a_track_time_list.append(a_track.time())

        # Find minimal value in time list
        min_time = min(a_track_time_list)

        # Find index of minimal value and corresponding track
        min_time_index = a_track_time_list.index(min_time)
        fastest_track = self.a_track_list[min_time_index]

        return fastest_track

    def shortest(self):
        """
        Find the track in the track collection with the shortest distance
        Returns
        -------
        object
            the SingleTrack object in the tracks collections with the shortest distance
        Examples
        --------
        >>> Tracks('short_tracks.json').shortest().cc
        '21144'
        """
        # Store list with distance for all tracks
        a_track_distance_list = []
        for a_track in self.a_track_list:
            a_track_distance_list.append(a_track.distance())

        # Find minimal value in distance list
        min_dist = min(a_track_distance_list)

        # Find index of minimal value and corresponding track
        min_dist_index = a_track_distance_list.index(min_dist)
        shortest_track = self.a_track_list[min_dist_index]

        return shortest_track

    def kmeans(self, clusters=3, iterations=10, bo=True):
        """Method which implements kmeans algorithm and takes number of
        desired clusters and iterations as an arguments.
        Parameters
        ---------
        clusters: int
            the required number of clusters
        iterations: int
            the number of iteration times
        bo: boolean
            Ture to call cluster function with numpy, False to call cluster function without numpy
        Return
        ------
        alloc: list
            a list with the track indices that belong to each group
        Example1
        ----------
        >>> len(Counter(Tracks('short_tracks.json').kmeans(clusters=2, iterations=50, bo=False)))
        2

        Example2
        ----------
        >>> len(Counter(Tracks('short_tracks.json').kmeans(clusters=2, iterations=50, bo=True)))
        2
        """

        if not isinstance(clusters, int):
            raise TypeError("Clusters = {} must be a integer".format(clusters))
        if not 0 < int(clusters) < self.__len__():
            raise ValueError("Clusters = {} must be bigger than 0 and less than length of tracks".format(clusters))

        if not isinstance(iterations, int):
            raise TypeError("iterations = {} must be a integer".format(iterations))
        if not 0 < int(iterations):
            raise ValueError("iterations = {} must be bigger than 0 ".format(iterations))

        co2_list = []
        distance_list = []
        time_list = []
        for i in range(len(self.a_track_list)):
            co2_list.append(self.a_track_list[i].co2())
            distance_list.append(self.a_track_list[i].distance())
            time_list.append(self.a_track_list[i].time())

        # get the tuple (co2, distance, time)
        target_tuple = list(zip(co2_list, distance_list, time_list))

        if not bo:
            # use cluster function without numpy
            alloc = cluster(target_tuple, clusters, iterations)
        else:
            # use cluster function with numpy
            alloc = cluster_numpy(target_tuple, clusters, iterations)

        return alloc

    def get_track(self, x):
        """
        Find the track in the track collection by its track number

        Parameters
        ----------
        x : int
            track number of tracks collection
            A positive integer which indicate the track number
        Returns
        ----------
        object
            A object of class SingleTrack
        Examples
        ----------
        >>> tracks = Tracks('short_tracks.json')
        >>> tracks.get_track(1).cc
        '11233344111'
        """

        if x > self.tracks_length:
            raise ValueError("the index x = {} must less than the length of tracks".format(x))
        if x < 0:
            raise ValueError("the index x = {} must be positive".format(x))

        return self.a_track_list[x - 1]


class SingleTrack:
    """
    Collect information of a single track and calculate the co2 consumption, total distance, time consumption and etc.
    """

    def __init__(self, start, cc, elevation, road, terrain):
        """
        Parameters
        ----------
        start: tuple
            the coordinate of the start
        cc: str
            the chaincode of the track
        elevation: list
            a list of heights along the track
        road: str
            the road type string of the track
        terrain: str
            the terrain type string of the track
        """

        self.track_len = len(cc) + 1
        self.start = start
        self.cc = cc
        self.elevation = elevation
        self.road = road
        self.terrain = terrain

    def __str__(self):
        """ __str__ shows the information of the track when we print the class object

        Returns
        ----------
        str
            the information of the track when we print the class object
        """
        return "<SingleTrack: starts at ({0}, {1}) - {2} steps>".format(self.start[0], self.start[1],
                                                                        self.track_len - 1)

    def __len__(self):
        """ __len__ shows the total number of coordinates in the track when we print len() of the class object

        Returns
        ----------
        int
            the total number of coordinates in the track when we print len() of the class object
        """
        return self.track_len

    def corners(self):
        """ corners function returns a list with coordinates of all the corners along the track

        Returns
        ----------
        list
            a list with coordinates of all the corners along the track

        Examples
        ----------
        >>> Tracks("short_tracks.json").get_track(4).corners()
        [(2, 3), (2, 4), (4, 4), (4, 2)]

        """

        freeman_cc2coord = {1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, -1)}
        cc = self.cc
        coordinate = self.start
        coordinates_list = []
        coordinates_list.append(coordinate)

        for i in range(len(cc)):
            coordinate = tuple(map(sum, zip(coordinate, freeman_cc2coord[int(cc[i])])))
            if i == (len(cc) - 1):
                coordinates_list.append(coordinate)
            else:
                if cc[i] != cc[i + 1]:
                    coordinates_list.append(coordinate)

        return coordinates_list

    def path(self):
        """ path function returns the text script for the path navigation

        Returns
        ----------
        list
            a list with all the text script for the path navigation

        """

        corners = self.corners()
        scripts_list = []

        for i in range(len(corners)):
            if i == 0:
                scripts_list.append("Start from {}".format(corners[i]))
            else:
                corners_diff = tuple(map(sub, corners[i], corners[i - 1]))
                if i != len(corners) - 1:
                    corners_diff_2 = tuple(map(sub, corners[i + 1], corners[i]))
                    if corners_diff_2[0] == 0:
                        if corners_diff_2[1] > 0:
                            if corners_diff[1] == 0:
                                if corners_diff[0] > 0:
                                    script = "Go east for {} km,".format(corners_diff[0]) \
                                             + " turn left at {}".format(corners[i])
                                    scripts_list.append(script)
                                if corners_diff[0] < 0:
                                    script = "Go west for {} km,".format(abs(corners_diff[0])) \
                                             + " turn right at {}".format(corners[i])
                                    scripts_list.append(script)
                        if corners_diff_2[1] < 0:
                            if corners_diff[1] == 0:
                                if corners_diff[0] > 0:
                                    script = "Go east for {} km,".format(corners_diff[0]) \
                                             + " turn right at {}".format(corners[i])
                                    scripts_list.append(script)
                                if corners_diff[0] < 0:
                                    script = "Go west for {} km,".format(abs(corners_diff[0])) \
                                             + " turn left at {}".format(corners[i])
                                    scripts_list.append(script)
                    if corners_diff_2[1] == 0:
                        if corners_diff_2[0] > 0:
                            if corners_diff[0] == 0:
                                if corners_diff[1] > 0:
                                    script = "Go north for {} km,".format(corners_diff[1]) \
                                             + " turn right at {}".format(corners[i])
                                    scripts_list.append(script)
                                if corners_diff[1] < 0:
                                    script = "Go south for {} km,".format(abs(corners_diff[1])) \
                                             + " turn left at {}".format(corners[i])
                                    scripts_list.append(script)
                        if corners_diff_2[0] < 0:
                            if corners_diff[0] == 0:
                                if corners_diff[1] > 0:
                                    script = "Go north for {} km,".format(corners_diff[1]) \
                                             + " turn left at {}".format(corners[i])
                                    scripts_list.append(script)
                                if corners_diff[1] < 0:
                                    script = "Go south for {} km,".format(abs(corners_diff[1])) \
                                             + " turn right at {}".format(corners[i])
                                    scripts_list.append(script)

                if i == len(corners) - 1:
                    if corners_diff[1] == 0:
                        if corners_diff[0] > 0:
                            script = "Go east for {} km,".format(corners_diff[0])
                            scripts_list.append(script)
                        if corners_diff[0] < 0:
                            script = "Go west for {} km,".format(abs(corners_diff[0]))
                            scripts_list.append(script)

                        if corners_diff[0] == 0:
                            if corners_diff[1] > 0:
                                script = "Go north for {} km,".format(corners_diff[1])
                                scripts_list.append(script)
                            if corners_diff[1] < 0:
                                script = "Go south for {} km,".format(abs(corners_diff[1]))
                                scripts_list.append(script)

                    scripts_list.append("reach your estination at {}".format(corners[i]))

        return scripts_list

    def visualise(self, show=True, filename="my_track.png"):
        """ visualise function shows and saves a graph with a distance vs elevation plot on the left
        and the coordinates of the path followed on the right

        Parameters
        ----------
        show: bool
            The graph will not show if show = False, defaults to True
        filename: str
            The graph will not be saved if the filename is changed, defaults to "my_track.png"

        Returns
        ----------
        file
            a graph with a distance vs elevation plot on the left
        """
        elevation_list = self.elevation
        distance_list = []
        distance_list.append(0)
        distance = 0

        for i in range(len(elevation_list)):
            if i != 0:
                distance = distance + sqrt(1 + (elevation_list[i] - elevation_list[i - 1]) ** 2)
                distance_list.append(distance)

        freeman_cc2coord = {1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, -1)}
        cc = self.cc
        coordinate = self.start
        x_list = []
        y_list = []
        x_list.append(coordinate[0])
        y_list.append(coordinate[1])

        for i in range(len(cc)):
            coordinate = tuple(map(sum, zip(coordinate, freeman_cc2coord[int(cc[i])])))
            x_list.append(coordinate[0])
            y_list.append(coordinate[1])

        # Plot the graphs side by side
        plt.subplot(1, 2, 1)
        plt.plot(distance_list, elevation_list, color='b', label="distance vs elevation graph")
        plt.ylabel('elevation')
        plt.xlabel('distance')

        plt.subplot(1, 2, 2)
        plt.plot(x_list, y_list, color='b', label="coordinates of the track")

        if filename == "my_track.png":
            plt.savefig(filename)
        if show:
            plt.show()

    def co2(self):
        """ co2 function returns the total co2 consumption of the track, unit of co2 consumption is kilogram

        Returns
        ----------
        float
            the total co2 consumption of the track

        Examples
        ----------
        >>> round(Tracks("short_tracks.json").get_track(4).co2(), 4)
        1.0065
        """

        elevation_list = self.elevation
        road = self.road
        terrain = self.terrain

        road_class = {'r': 1.4, 'l': 1, 'm': 1.25}
        terrain_class = {'d': 2.5, 'g': 1.25, 'p': 1}

        co2_consumption = 0

        for i in range(len(elevation_list)):
            if i != 0:
                distance = sqrt(1 + ((elevation_list[i] - elevation_list[i - 1]) / 1000) ** 2)
                slope = ((elevation_list[i] - elevation_list[i - 1]) / 1000) * 100
                road_factor = road_class[road[i - 1]]
                terrain_factor = terrain_class[terrain[i - 1]]

                if slope >= -2 and slope <= 2:
                    slope_factor = 1

                elif slope > 2 and slope <= 6:
                    slope_factor = 1.3

                elif slope > 6 and slope <= 10:
                    slope_factor = 2.35

                elif slope > 10:
                    slope_factor = 2.9

                elif slope >= -6 and slope < -2:
                    slope_factor = 0.45

                elif slope >= -10 and slope < -6:
                    slope_factor = 0.16

                # According to the equation given in the pdf, we can work out the total co2 consumption
                co2_consumption = co2_consumption + (5.4 / 100) * slope_factor * road_factor * terrain_factor \
                                  * distance * 2.6391

        return co2_consumption

    def time(self):
        """ time function returns the total time needed for the track, unit of time is hour

        Returns
        ----------
        float
            the total time needed for the track

        Examples
        ----------
        >>> round(Tracks("short_tracks.json").get_track(4).time(), 4)
        0.0708
        """

        elevation_list = self.elevation
        road = self.road

        road_class = {'r': 30, 'l': 80, 'm': 120}

        total_time = 0

        for i in range(len(elevation_list)):
            if i != 0:
                distance = sqrt(1 + ((elevation_list[i] - elevation_list[i - 1]) / 1000) ** 2)
                road_speed = road_class[road[i - 1]]

                total_time = total_time + distance / road_speed

        return total_time

    def distance(self):
        """ distance function returns the total distance of the track, unit of distance is kilometre

        Returns
        ----------
        float
            the total distance of the track

        Examples
        ----------
        >>> round(Tracks("short_tracks.json").get_track(4).distance(), 6)
        5.000038
        """
        elevation_list = self.elevation
        distance = 0

        for i in range(len(elevation_list)):
            if i != 0:
                distance = distance + sqrt(1 + ((elevation_list[i] - elevation_list[i - 1]) / 1000) ** 2)

        return distance


if __name__ == "__main__":
    import doctest

    # doctest.testmod()

    parser = ArgumentParser(description="read greentrack from command line")
    parser.add_argument('--start', help="Input start coordinates", nargs=2, type=int)
    parser.add_argument('--end', help="Input end coordinates", nargs=2, type=int)
    parser.add_argument("--verbose", nargs="?", const="True")
    arguments = parser.parse_args()

    # Calling the function
    tracks = query_tracks(arguments.start, arguments.end, min_steps_straight=1, max_steps_straight=6, n_tracks=30,
                          save=True)
    print(tracks)
    greenest = tracks.greenest()
    path = greenest.corners()
    path_with_format = [str(i) for i in path]
    path_script = greenest.path()
    co2 = greenest.co2()

    # Transfer the time value to the given format
    time = greenest.time()

    for i in range(len(str(time))):
        if str(time)[i] == ".":
            index_1 = i

    if index_1 != 1:
        time_h = int(str(time)[:(index_1 - 1)])
    else:
        time_h = int(str(time)[index_1 - 1])

    if len(str(time_h)) < 2:
        time_h = "0{}".format(time_h)

    if index_1 + 1 != len(str(time)) - 1:
        time_min = float("0.{}".format(str(time)[(index_1 + 1):])) * 60
    else:
        time_min = float("0.{}".format(str(time)[index_1 + 1])) * 60

    for i2 in range(len(str(time_min))):
        if str(time_min)[i2] == ".":
            index_2 = i2

    if index_2 != 1:
        time_min_2 = int(str(time_min)[:(index_2 - 1)])
    else:
        time_min_2 = int(str(time_min)[index_2 - 1])

    if len(str(time_min_2)) < 2:
        time_min_2 = "0{}".format(time_min_2)

    if index_2 + 1 != len(str(time_min)) - 1:
        time_s = round(float("0.{}".format(str(time_min)[(index_2 + 1):])) * 60)
    else:
        time_s = round(float("0.{}".format(str(time_min)[index_2 + 1])) * 60)

    if len(str(time_s)) < 2:
        time_s = "0{}".format(time_s)

    time_with_format = "{}:{}:{}".format(time_h, time_min_2, time_s)

    # Print the results
    if not arguments.verbose:
        print("Path: {}".format(", ".join(path_with_format)))

    if arguments.verbose:
        print("Path:")
        for i in path_script:
            print("- {}".format(i))

    print("CO2: {} kg".format(round(co2, 2)))
    print("Time: {}".format(time_with_format))