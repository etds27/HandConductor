import logging
import math
import sys

import cv2
import numpy as np

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

LANDMARK_MAP = {
    "WRIST": 0,
    "THUMB_CMC": 1,
    "THUMB_MCP": 2,
    "THUMB_IP": 3,
    "THUMB_TIP": 4,
    "INDEX_FINGER_CMC": 5,
    "INDEX_FINGER_PIP": 6,
    "INDEX_FINGER_DIP": 7,
    "INDEX_FINGER_TIP": 8,
    "MIDDLE_FINGER_CMC": 9,
    "MIDDLE_FINGER_PIP": 10,
    "MIDDLE_FINGER_DIP": 11,
    "MIDDLE_FINGER_TIP": 12,
    "RING_FINGER_CMC": 13,
    "RING_FINGER_PIP": 14,
    "RING_FINGER_DIP": 15,
    "RING_FINGER_TIP": 16,
    "PINKY_CMC": 17,
    "PINKY_PIP": 18,
    "PINKY_DIP": 19,
    "PINKY_TIP": 20
}

palm_box_landmark_indices = np.array([LANDMARK_MAP["WRIST"],
                                      LANDMARK_MAP["THUMB_CMC"],
                                      LANDMARK_MAP["INDEX_FINGER_CMC"],
                                      LANDMARK_MAP["MIDDLE_FINGER_CMC"],
                                      LANDMARK_MAP["RING_FINGER_CMC"],
                                      LANDMARK_MAP["PINKY_CMC"]])

finger_linkage_indices = np.arange(1, 21).reshape((5, 4))


class HandController:
    def __init__(self, hand_landmarks, image=np.array([[0]])):
        self.hand_angle = 0
        self.hand_landmarks = hand_landmarks  # This is the object that contains the collection of landmark coords
        self.landmarks = hand_landmarks.landmark
        self.image = image

    def normalize_to_image(self, point):
        return np.multiply(point, np.flip(self.image.shape[:2])).astype('int32')

    def determine_hand_orientation(self):
        """
        This originally meant to be used to find an angle to rotate the axis about.
        The new axis would be used to determine if fingers were up or down.
        Instead, the points were converted to polar. Keeping this in for future use

        Determine a rough angle of hand orienation. This will be done by averaging the angle between the THUMB_CMC and the INDEX_FINGER_CMC and the angle between the WRIST and PINKY_CMC.
        Only use x, y for angle.
        Assume hand is normal to camera

        :param landmarks:
        :return: angle at which to reorient axes by. (in radians)
        """
        t_cmc = np.array([self.landmarks[LANDMARK_MAP["THUMB_CMC"]].x, self.landmarks[LANDMARK_MAP["THUMB_CMC"]].y])
        i_cmc = np.array(
            [self.landmarks[LANDMARK_MAP["INDEX_FINGER_CMC"]].x, self.landmarks[LANDMARK_MAP["INDEX_FINGER_CMC"]].y])

        p_cmc = np.array([self.landmarks[LANDMARK_MAP["PINKY_CMC"]].x, self.landmarks[LANDMARK_MAP["PINKY_CMC"]].y])
        wrist = np.array([self.landmarks[LANDMARK_MAP["WRIST"]].x, self.landmarks[LANDMARK_MAP["WRIST"]].y])

        # temp = [[x1, y1], [x2, y2]]
        temp = np.array([i_cmc - t_cmc, p_cmc - wrist])
        # temp = [[x1, x2], [y1, y2]]
        temp = np.transpose(temp)

        # y array as 1st arg, x array as second
        temp = np.arctan2(temp[1, :], temp[0, :])
        return temp.mean()

    def draw_orientation_angle(self, image, angle):
        """
        Draws a line stemming from halfway point between wrist and thumb cmc, outward in the direction of the average
            angle between the points of thumb_cmc to index_cmc, and the wrist to pinky_cmc


        :param image:
        :param landmarks: collection of all landmarks
        :param angle: in rad
        :return:
        """
        t_cmc = np.array([self.landmarks[LANDMARK_MAP["THUMB_CMC"]].x, self.landmarks[LANDMARK_MAP["THUMB_CMC"]].y])
        wrist = np.array([self.landmarks[LANDMARK_MAP["WRIST"]].x, self.landmarks[LANDMARK_MAP["WRIST"]].y])

        # halfway between the two points
        start_point = ((t_cmc + wrist) / 2)

        # normalize the landmark points to screen size
        start_point = np.multiply(start_point, np.flip(image.shape[:2])).astype('int32')

        x_offset = math.cos(angle) * 1000
        y_offset = math.sin(angle) * 1000
        # print(x_offset, y_offset, angle * 180 / np.pi)

        end_point = np.array([x_offset + start_point[0], y_offset + start_point[1]], 'int32')

        # Minor discontiuties around pi / 2. Near left portion of screen
        cv2.line(image, start_point, end_point, (0, 0, 0), 4)

        cv2.putText(image, "%i deg" % (angle * 180 / np.pi), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
                    cv2.LINE_AA)

    def _find_center_of_hand(self):
        poi_coords = np.array([[self.landmarks[i].x, self.landmarks[i].y] for i in [
            LANDMARK_MAP["WRIST"],
            LANDMARK_MAP["THUMB_CMC"],
            LANDMARK_MAP["INDEX_FINGER_CMC"],
            LANDMARK_MAP["PINKY_CMC"],
        ]])
        return poi_coords.mean(0)

    def convert_hand_to_polar(self):
        """
        Converts the current hand landmarks to polar coordinates where the center is the center of the palm
        :param landmarks:
        :return:
        """
        center = self._find_center_of_hand()

        polar_landmarks = np.zeros((len(LANDMARK_MAP.keys()), 2))
        for i, landmark in enumerate(self.landmarks):
            x, y = landmark.x - center[0], landmark.y - center[1]
            rad = np.sqrt(x ** 2 + y ** 2)
            deg = np.arctan2(y, x)
            polar_landmarks[i, :] = np.array([rad, deg])

        return polar_landmarks, center

    def draw_hand_range_rings(self, center, resolution=50, end=1000):
        image = cv2.circle(self.image, center, 1, (0, 0, 0), 2)
        for radius in range(resolution, end + 1, resolution):
            image = cv2.circle(image, center, radius, (0, 0, 0), 2)
        return image

    def draw_palm_fill(self):
        """
        Draws a filled polygon within the following points.
        :return:
        """
        cv2.fillConvexPoly(self.image,
                           points=np.array([[int(self.landmarks[i].x * np.shape(self.image)[1]),
                                             int(self.landmarks[i].y * np.shape(self.image)[0])]
                                            for i in palm_box_landmark_indices], dtype=np.int32),
                           color=(255, 0, 0))

    finger_linkage_indices = np.arange(1, 21).reshape((5, 4))

    def get_fingers_up(self):
        """
        Determines the fingers that are up.
        Converts cartesian landmark coordinates to polar with the center of the palm acting as the reference point
        Then checks all landmarks for each fingers
        If the magnitude of all landmarks for a finger are in ascending order, then the finger is up

        :return: an np array that acts as a mask for which fingers are up or down. [thumb, index, middle, ring, pinky]
        """
        polar_landmarks, center = self.convert_hand_to_polar()
        finger_up = np.zeros((1, np.shape(finger_linkage_indices)[0]))
        for finger_number, indices in enumerate(finger_linkage_indices):

            # Thumb is difficult to discern if it is up or down based on current algorithm
            # More friendly algorithm would be to determine the angle between linkages
            if finger_number == 0:
                finger_up[0][0] = 0
                continue

            temp = -sys.maxsize
            up = True
            for index in indices:
                r = polar_landmarks[index][0]
                if r < temp:
                    up = False
                    break
                temp = r
            if up:
                finger_up[0, finger_number] = 1
        return finger_up

    def get_num_fingers_up(self):
        """

        :return: the number of fingers held up as an integer
        """
        return int(np.sum(self.get_fingers_up()))

    def create_image(self, image=None):
        """
        Draws hand graphics on the image
        :param image:
        :return: image
        """
        if image is not None:
            self.image = image

        mp_drawing.draw_landmarks(
            image,
            self.hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # angle = determine_hand_orientation(hand_landmarks.landmark)
        # draw_orientation_angle(image, hand_landmarks.landmark, angle)

        for i, landmark in enumerate(self.landmarks):
            x, y = int(landmark.x * np.shape(image)[1]), int(landmark.y * np.shape(image)[0])
            image = cv2.putText(image, "%i" % i, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        image = self.draw_hand_range_rings(self.normalize_to_image(self._find_center_of_hand()), end=150)

        return image


