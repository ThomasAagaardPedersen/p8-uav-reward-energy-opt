import setup_path
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2

import asyncio
import tornado.platform.asyncio

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True)

state = client.getMultirotorState()
s = pprint.pformat(state)
print("state: %s" % s)

imu_data = client.getImuData()
s = pprint.pformat(imu_data)
print("imu_data: %s" % s)

barometer_data = client.getBarometerData()
s = pprint.pformat(barometer_data)
print("barometer_data: %s" % s)

magnetometer_data = client.getMagnetometerData()
s = pprint.pformat(magnetometer_data)
print("magnetometer_data: %s" % s)

gps_data = client.getGpsData()
s = pprint.pformat(gps_data)
print("gps_data: %s" % s)

airsim.wait_key("Press any key to takeoff")
print("Taking off...")
client.armDisarm(True)
client.takeoffAsync().join()

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

# Main movement chunk

# airsim.wait_key('Press any key to move vehicle to (-10, 10, -10) at 5 m/s')
# client.moveToPositionAsync(-10, 10, -10, 5).join()

airsim.wait_key("Press any key to move to the right of the windturbine at 5 m/s")
# client.moveToPositionAsync(33637, 332, 6,992, 5).join()
client.moveToPositionAsync(0, 100, 0, 4).join()


airsim.wait_key("Press any key to move to the left of the windturbine at 5 m/s")
# client.moveToPositionAsync(33637, 332, 6,992, 5).join()
client.moveToPositionAsync(0, 100, -60, 4)
airsim.wait_key("Press any key to move to the left of the windturbine at 5 m/s")
client.moveToPositionAsync(0, -100, -60, 4)
airsim.wait_key("Press any key to move to the left of the windturbine at 5 m/s")
client.moveToPositionAsync(0, -100, 20, 4)
airsim.wait_key("Press any key to move to the left of the windturbine at 5 m/s")
client.moveToPositionAsync(0, 100, 20, 4)

# Position reset

airsim.wait_key("Press any key to reset to original state")

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
