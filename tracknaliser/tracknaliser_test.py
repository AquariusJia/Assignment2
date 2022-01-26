import unittest
import pytest
from tracknaliser import *
from unittest.mock import patch


class tracknaliserTest(unittest.TestCase):
    def setUp(self):
        test_tracks = {
            "metadata": {
                "datetime": "2021-12-11T21:12:20",
                "end": [
                    4,
                    2
                ],
                "mapsize": [
                    5,
                    5
                ],
                "n_tracks": 5,
                "rangesteps": [
                    1,
                    2
                ],
                "resolution": 1,
                "start": [
                    2,
                    3
                ],
                "units_elevation": "m",
                "units_steps": "km"
            },
            "tracks": [
                {
                    "cc": "11233344111",
                    "elevation": [
                        17,
                        18,
                        19,
                        24,
                        23,
                        22,
                        21,
                        16,
                        11,
                        12,
                        13,
                        14
                    ],
                    "road": "llmmmmlrrrr",
                    "terrain": "pggppdddppg"
                },
                {
                    "cc": "443411122",
                    "elevation": [
                        17,
                        12,
                        7,
                        6,
                        1,
                        2,
                        3,
                        4,
                        9,
                        14
                    ],
                    "road": "rrrrrrrrr",
                    "terrain": "ppddppggg"
                },
                {
                    "cc": "3341111",
                    "elevation": [
                        17,
                        16,
                        15,
                        10,
                        11,
                        12,
                        13,
                        14
                    ],
                    "road": "llrrrrr",
                    "terrain": "ddddppg"
                },
                {
                    "cc": "21144",
                    "elevation": [
                        17,
                        22,
                        23,
                        24,
                        19,
                        14
                    ],
                    "road": "mmmlr",
                    "terrain": "ppggg"
                },
                {
                    "cc": "343411121",
                    "elevation": [
                        17,
                        16,
                        11,
                        10,
                        5,
                        6,
                        7,
                        8,
                        13,
                        14
                    ],
                    "road": "lrrrrrrrr",
                    "terrain": "dddddpppg"
                }
            ]
        }
        track_filename = 'test_data.json'

        with open(track_filename, 'w') as f:
            json.dump(test_tracks, f)

        tracks = Tracks(track_filename)

        self.tracks = tracks

    # Tests for Tracks object
    def test_kmeans(self):
        # test the cluster function without numpy
        alloc = self.tracks.kmeans(2, 50, False)
        assert len(Counter(alloc)) == 2
        # test the cluster function with numpy
        alloc_np = self.tracks.kmeans(2, 50, True)
        assert int(len(Counter(alloc_np))) == 2

    def test_clusters_is_not_available(self):
        with pytest.raises(TypeError) as exception:
            self.tracks.kmeans('a', 10)
        assert str(exception.value) == "Clusters = {} must be a integer".format('a')

        with pytest.raises(ValueError) as exception:
            self.tracks.kmeans(10, 10)
        assert str(exception.value) == "Clusters = {} must be bigger than 0 and less than length of tracks".format(10)

    def test_iterations_is_not_available(self):
        with pytest.raises(TypeError) as exception:
            self.tracks.kmeans(3, 'b')
        assert str(exception.value) == "iterations = {} must be a integer".format('b')

        with pytest.raises(ValueError) as exception:
            self.tracks.kmeans(3, -1)
        assert str(exception.value) == "iterations = {} must be bigger than 0 ".format(-1)

    def test_greenest(self):
        assert self.tracks.greenest().cc == "21144"

    def test_fastest(self):
        assert self.tracks.fastest().cc == "21144"

    def test_shortest(self):
        assert self.tracks.shortest().cc == "21144"

    def test_track_number_beyond_length_of_tracks(self):
        with pytest.raises(ValueError) as exception:
            self.tracks.get_track(10)
        assert str(exception.value) == "the index x = {} must less than the length of tracks".format(10)

    def test_track_number_is_negative(self):
        with pytest.raises(ValueError) as exception:
            self.tracks.get_track(-1)
        assert str(exception.value) == "the index x = {} must be positive".format(-1)

    def test_track(self):
        assert self.tracks.get_track(1).cc == "11233344111"
        assert self.tracks.get_track(2).cc == "443411122"

    # Tests for SingleTrack object
    def test_inputs(self):
        self.assertIsInstance(self.tracks.start, tuple)
        self.assertIsInstance(self.tracks.get_track(4).cc, str)
        self.assertIsInstance(self.tracks.get_track(4).elevation, list)
        self.assertIsInstance(self.tracks.get_track(4).road, str)
        self.assertNotIsInstance(self.tracks.get_track(4).terrain, list)

    def test_corners(self):
        expectCorners = [(2, 3), (2, 4), (4, 4), (4, 2)]
        realCorners = self.tracks.get_track(4).corners()
        message = "Corners coordinates are not correct"

        self.assertEqual(expectCorners, realCorners, message)

    def test_co2(self):
        expectConsumption = 1.0065
        realConsumption = round(self.tracks.get_track(4).co2(), 4)
        message = "CO2 consumption is not correct"

        self.assertEqual(expectConsumption, realConsumption, message)

    def test_time(self):
        expectTime = 0.0708
        realTime = round(self.tracks.get_track(4).time(), 4)
        message = "Time is not correct"

        self.assertEqual(expectTime, realTime, message)

    def test_distance(self):
        expectDistance = 5.000038
        realDistance = round(self.tracks.get_track(4).distance(), 6)
        message = "Distance is not correct"

        self.assertEqual(expectDistance, realDistance, message)


class QueryLoadData(unittest.TestCase):
    def test_illegal_chaincode(self):
        test_negative_tracks_data = {
            "metadata": {
                "datetime": "2021-12-11T21:12:20",
                "end": [
                    4,
                    2
                ],
                "mapsize": [
                    5,
                    5
                ],
                "n_tracks": 5,
                "rangesteps": [
                    1,
                    2
                ],
                "resolution": 1,
                "start": [
                    2,
                    3
                ],
                "units_elevation": "m",
                "units_steps": "km"
            },
            "tracks": [
                {
                    "cc": "911233344111",
                    "elevation": [
                        17,
                        18,
                        19,
                        24,
                        23,
                        22,
                        21,
                        16,
                        11,
                        12,
                        13,
                        14
                    ],
                    "road": "llmmmmlrrrr",
                    "terrain": "pggppdddppg"
                }]
        }
        with open('test_negative_tracks_data.json', 'w') as f:
            json.dump(test_negative_tracks_data, f)
        with pytest.raises(ValueError) as exception:
            load_tracksfile('test_negative_tracks_data.json')
        assert str(exception.value) == "The chaincode digits must be from 1 to 4"

    def test_illegal_roda_type(self):
        test_negative_tracks_data = {
            "metadata": {
                "datetime": "2021-12-11T21:12:20",
                "end": [
                    4,
                    2
                ],
                "mapsize": [
                    5,
                    5
                ],
                "n_tracks": 5,
                "rangesteps": [
                    1,
                    2
                ],
                "resolution": 1,
                "start": [
                    2,
                    3
                ],
                "units_elevation": "m",
                "units_steps": "km"
            },
            "tracks": [
                {
                    "cc": "11233344111",
                    "elevation": [
                        17,
                        18,
                        19,
                        24,
                        23,
                        22,
                        21,
                        16,
                        11,
                        12,
                        13,
                        14
                    ],
                    "road": "llmmmmlrrrrkk",
                    "terrain": "pggppdddppg"
                }]
        }
        with open('test_negative_tracks_data.json', 'w') as f:
            json.dump(test_negative_tracks_data, f)
        with pytest.raises(ValueError) as exception:
            load_tracksfile('test_negative_tracks_data.json')
        assert str(exception.value) == "The rode-type code must be r, l, or m"

    def test_illegal_terrain_type(self):
        test_negative_tracks_data = {
            "metadata": {
                "datetime": "2021-12-11T21:12:20",
                "end": [
                    4,
                    2
                ],
                "mapsize": [
                    5,
                    5
                ],
                "n_tracks": 5,
                "rangesteps": [
                    1,
                    2
                ],
                "resolution": 1,
                "start": [
                    2,
                    3
                ],
                "units_elevation": "m",
                "units_steps": "km"
            },
            "tracks": [
                {
                    "cc": "11233344111",
                    "elevation": [
                        17,
                        18,
                        19,
                        24,
                        23,
                        22,
                        21,
                        16,
                        11,
                        12,
                        13,
                        14
                    ],
                    "road": "llmmmmlrrrr",
                    "terrain": "pggppdddppgkk"
                }]
        }
        with open('test_negative_tracks_data.json', 'w') as f:
            json.dump(test_negative_tracks_data, f)
        with pytest.raises(ValueError) as exception:
            load_tracksfile('test_negative_tracks_data.json')
        assert str(exception.value) == "The rode-type code must be d, g, or p"

    def test_illegal_start_point(self):
        with pytest.raises(ValueError) as exception:
            query_tracks(start=[-1, 0], save=False)
        assert str(exception.value) == "The queried coordinates {} should not be negative".format([-1, 0])

    def test_illegal_end_point(self):
        with pytest.raises(ValueError) as exception:
            query_tracks(end=[-1, 0], save=False)
        assert str(exception.value) == "The queried coordinates {} should not be negative".format([-1, 0])

    def test_illegal_min_steps(self):
        with pytest.raises(ValueError) as exception:
            query_tracks(min_steps_straight=-1 , save=False)
        assert str(exception.value) == "The minimum steps straight value cannot be less than 1"

    def test_illegal_min_steps_max_steps(self):
        with pytest.raises(ValueError) as exception:
            query_tracks(min_steps_straight=6, max_steps_straight=1, save=False)
        assert str(exception.value) == "The maximum steps straight value cannot be less than the minimum steps straight value"

    def test_illegal_n_tracks(self):
        with pytest.raises(ValueError) as exception:
            query_tracks(n_tracks=0, save=False)
        assert str(exception.value) == "The number of tracks cannot be less than 1"

    def test_build_default_params(self):
        start = [0, 0]
        end = [299, 299]
        min_steps_straight = 1
        max_steps_straight = 6
        n_tracks = 300
        save = False
        with patch.object(requests, 'get') as mock_get:
            default_tracks = query_tracks(start, end, min_steps_straight,
                                         max_steps_straight, n_tracks, save)
            mock_get.assert_called_with(
                "https://ucl-rse-with-python.herokuapp.com/road-tracks/tracks/?start_point_x={}&start_point_y={}&end_point_x={}&end_point_y={}&min_steps_straight={}&max_steps_straight={}&n_tracks={}".format(
                    start[0], start[1], end[0], end[1], min_steps_straight, max_steps_straight, n_tracks)
            )


if __name__ == '__main__':
    unittest.main()