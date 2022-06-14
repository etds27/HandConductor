#!/usr/bin/env python3
import json
import logging

import subprocess
import threading
import multiprocessing
import time
import colorsys

BULB_IP = "192.168.0.131"

FINGERS_TO_HEX = {
    1: "#FF0000",
    2: "#00FF00",
    3: "#0000FF",
    4: "#00FFFF"
}

FINGERS_TO_HSV = {
    1: {"HUE": 0, "SATURATION": 100, "VALUE": 100},
    2: {"HUE": 120, "SATURATION": 100, "VALUE": 100},
    3: {"HUE": 240, "SATURATION": 100, "VALUE": 100},
    4: {"HUE": 300, "SATURATION": 100, "VALUE": 100},
}

logging.basicConfig(level=logging.DEBUG)


def rgb_to_hsv(r, g, b):
    pass


class LightController:
    """
    class responsible for controlling the behavior of the TPLINK LB130

    currently works relatively well without issue. Can run into problems when sending commands too fast.
    The bulb doesnt seem to be able to respond quickly and causes thread timeouts.
    No workaround for this issue
    """
    def __init__(self):
        self.process = None
        pass

    def get_available_devices(self):
        """
        TODO:
        Used to find all TP LINK devices on current network
        :return: JSON that maps device name to device type and IP
        """
        logging.info("Scanning for TP devices")
        p = subprocess.Popen("tplight scan", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print(p.stderr.read())
        print(p.stdout.readline())
        print(p.stdout.readline())

    def set_color_by_finger(self, num):
        """
        Creates the thread to send the npm command to the bulb
        :param num: int: 0 = off, 1 - n = mapped value in FINGERS_TO_HEX
        :return:
        """
        logging.info("LightController.set_color_by_finger: num = %i" % num)
        logging.debug("LightController.set_color_by_finger: active threads %i" % threading.active_count())

        # If 0 was given, turn off bulb
        if num == 0:
            if self.process is None or not self.process.is_alive():
                logging.info("LightController.set_color_by_finger: creating new turn off thread")
                self.process = ProcessThread("tplight off %s" % BULB_IP)
                logging.info("LightController.set_color_by_finger: thread created")
                self.process.start()
        else:
            if num in FINGERS_TO_HEX:
                logging.info(str(self.process))
                if self.process is not None:
                    logging.debug("LightController.set_color_by_finger: Previous thread is still alive: %s" % self.process.is_alive())

                logging.info("LightController.set_color_by_finger: creating new thread")

                """
                tplight raw 192.168.0.131 '{"smartlife.iot.smartbulb.lightingservice":{"transition_light_state":{"ignore_default":1,"on_off":1,"color_temp":0,"hue":100,"saturation":100}}}'
                """
                self.process = ProcessThread("tplight hex %s \"%s\"" % (BULB_IP, FINGERS_TO_HEX[num]))
                self.process.start()
                # raw = '{"smartlife.iot.smartbulb.lightingservice":{"transition_light_state":{"ignore_default":1,"on_off":1,"color_temp":0,"hue":%i,"saturation":%i,"brightness":%i}}}' % (FINGERS_TO_HSV[num]["HUE"], FINGERS_TO_HSV[num]["SATURATION"], FINGERS_TO_HSV[num]["VALUE"])
                # self.process = ProcessThread("tplight raw %s '%s'" % (BULB_IP, raw))
                logging.debug("LightController.set_color_by_finger: thread started")
        # self.process.join(0.25)


class ProcessThread(threading.Thread):
    def __init__(self, cmd):
        super(ProcessThread, self).__init__()
        self.cmd = cmd

    def run(self):
        logging.info("ProcessThread.run: sending command '%s'" % self.cmd)
        """
        stdout, stderr = subprocess.Popen(self.cmd, shell=True,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE).communicate()
        """
        p = subprocess.Popen(self.cmd, shell=True,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
        p.wait(0.2)
        logging.debug(p.stdout.readlines())
        logging.debug(p.stderr.readlines())


if __name__ == "__main__":
    lc = LightController()

    lc.set_color_by_finger(4)
    time.sleep(1)
    lc.set_color_by_finger(0)
    time.sleep(1)
    lc.set_color_by_finger(2)
    # time.sleep(5)
    # lc.turn_off()
