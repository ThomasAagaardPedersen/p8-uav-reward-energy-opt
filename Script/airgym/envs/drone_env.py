import csv
import math
import os
import sys
from argparse import Action, ArgumentParser
from time import perf_counter
from xml.etree.ElementTree import tostring
import time
import airsim
import cv2
import gym
import numpy as np
import openpyxl
import setup_path
import torch
from airgym.envs.airsim_env import AirSimEnv
from gym import spaces
from pathlib import Path

model_path = Path(__file__).parent.parent.parent / "path/to/best_WTB.pt"
# model_path = "D:/Unreal_Projects/P8/Script/path/to/best_WTB.pt"
# model_path = "D:/Unreal_Projects/P8/Script/path/to/bestv8.pt"

model = torch.hub.load(
    "ultralytics/yolov5", "custom", path=model_path, force_reload=True
)  #  local model
print("Model has been downloaded and created")

global curr_time, prev_time, detected, episode_length
curr_time = 0
prev_time = perf_counter()
detected = True
episode_length = 0

# Energy Consumption global variables
global prev_E, prev_E_time  # , hovering, horizontal, vertical_up, vertical_down, altitude, payload
prev_E = 1
prev_E_time = time.time()

"""
hovering = 0
horizontal = 0
vertical_up = 0
vertical_down = 0
altitude = 0
payload = 0
"""


class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address, step_length, image_shape, fog_level):
        super().__init__(image_shape)
        self.step_length = step_length
        self.image_shape = image_shape
        self.stepTime = 5
        self.fog_level = fog_level

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
            "orientation": np.zeros(3),
            "velocity": np.zeros(3),
        }

        self.cam_coords = {
            "xmin": 0,
            "ymin": 0,
            "xmax": 0,
            "ymax": 0,
            "height": 0,
            "width": 0,
            "confidence": 0,
        }
        self.edge_coords = {
            "edge_x1": 0,
            "edge_y1": 0,
            "edge_x2": 0,
            "edge_y2": 0,
        }

        self.depthDistance = 30.0
        self.prev_depthDistance = 100.0

        self.drone = airsim.MultirotorClient(ip=ip_address)

        self.totalPower = 0

        self.action_space = spaces.Discrete(
            6
        )  # Number of possible actions/movements in the action_space

        self._setup_flight()
        """ DEBUGGING
        init_camera_info = self.drone.simGetCameraInfo("high_res")
        print(type(init_camera_info))

        self.drone_state = self.drone.getMultirotorState()
        self.state["orientation"] = self.drone_state.kinematics_estimated.orientation
        print(self.state["orientation"])
        quaterion = self.state["orientation"]
        z = quaterion.z_val
        print(z)
        #camera_pose = airsim.Pose(airsim.Vector3r(self.state(position)), airsim.to_quaternion(0, 0, )
        """
        self.prev_x_size = 0
        self.prev_y_size = 0
        self.x_size = 0
        self.y_size = 0

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        # self.drone.moveToPositionAsync(-0.55265, -31.9786, -19.0225, 10).join()
        # self.drone.moveToPositionAsync(256, -4, -60, 10).join()
        # self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()
        self.drone.moveByVelocityAsync(0, 0, -0.8, 5).join()  # move 4 meters up

    def detectAndMark(self, image):
        result = model(image)
        is_detected = True
        objs = result.pandas().xyxy[0]
        objs_name = objs.loc[objs["name"] == "WTB"]
        height = image.shape[0]
        width = image.shape[1]
        x_middle = 0
        y_middle = 0
        x_min = 0
        y_min = 0
        x_max = 0
        y_max = 0
        conf = 0
        try:
            obj = objs_name.iloc[0]

            conf = obj.confidence
            x_min = obj.xmin
            y_min = obj.ymin
            x_max = obj.xmax
            y_max = obj.ymax
            x_middle = x_min + (x_max - x_min) / 2
            y_middle = y_min + (y_max - y_min) / 2

            x_middle = round(x_middle, 0)
            y_middle = round(y_middle, 0)
            # Calculate the distance from the middle of the camera frame view, to the middle of the object
            # x_distance = x_middle-width/2
            # y_distance = y_middle-height/2

            cv2.rectangle(
                image,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                (0, 255, 0),
                2,
            )
            cv2.circle(image, (int(x_middle), int(y_middle)), 5, (0, 255, 0), 2)
            cv2.circle(image, (int(width / 2), int(height / 2)), 5, (0, 0, 255), 2)
            cv2.line(
                image,
                (int(x_middle), int(y_middle)),
                (int(width / 2), int(height / 2)),
                (0, 0, 255),
                2,
            )
            cv2.line(
                image,
                (int((width / 2) - 200), int(0)),
                (int((width / 2) - 200), int(height)),
                (255, 0, 0),
                2,
            )
            cv2.line(
                image,
                (int((width / 2) + 200), int(0)),
                (int((width / 2) + 200), int(height)),
                (255, 0, 0),
                2,
            )
        except:
            print("Error")
            is_detected = False
        return image, x_min, y_min, x_max, y_max, width, height, is_detected, conf

    def edge_detection(self, depth_image):
        depth_image = cv2.convertScaleAbs(depth_image, alpha=255.0 / depth_image.max())
        # Convert to grayscale
        # gray = cv2.cvtColor(depth_image,cv2.COLOR_BGR2GRAY)
        # Apply Gausian blur
        blur = cv2.GaussianBlur(depth_image, (5, 5), 0)
        # Use canny edge detection
        edges = cv2.Canny(blur, 50, 150)
        # Apply HoughLinesP method to to directly obtain line end points
        lines_list = []
        lines = cv2.HoughLinesP(
            edges,  # Input edge image
            1,  # Distance resolution in pixels
            np.pi / 180,  # Angle resolution in radians
            threshold=70,  # Min number of votes for valid line
            minLineLength=30,  # Min allowed length of line
            maxLineGap=40,  # Max allowed gap between line for joining them
        )
        # Eliminate non-vertical lines
        non_ver_lines = []
        for points in lines:
            x1, y1, x2, y2 = points[0]
            if abs(x1 - x2) > 10:
                non_ver_lines.append([x1, y1, x2, y2])
        # Get the longest line
        longest_lines = []

        for points in non_ver_lines:
            x1, y1, x2, y2 = points
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            # if the length of line is greater than the length of current longest lines, add it to the longest lines list
            if len(longest_lines) < 2:
                longest_lines.append((points, length))
            else:
                for i, l in enumerate(longest_lines):
                    if length > l[1]:
                        longest_lines[i] = (points, length)
                        break
        # Iterate over points
        for points in longest_lines:
            (x1, y1, x2, y2), _ = points
            cv2.line(depth_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(depth_image, (x1, y1), 7, (255, 0, 0), 2)
            cv2.circle(depth_image, (x2, y2), 7, (255, 0, 0), 2)
            cv2.circle(
                depth_image,
                (round((x1 + x2) / 2), round((y1 + y2) / 2)),
                7,
                (255, 0, 0),
                2,
            )

        (x1, y1, x2, y2), _ = longest_lines[0]

        return depth_image, x1, y1, x2, y2

    def transform_obs(self, responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float64)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

    def _get_obs(self):
        global detected
        camera_type = "high_res"  # optionally: "high_res"
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        # Locking the camera orientation to pose of the drone and the orientation to (0,0,drone_yaw)
        self.state["orientation"] = self.drone_state.kinematics_estimated.orientation
        drone_orientation = self.state["orientation"]
        yaw_drone_frame = drone_orientation.z_val
        x_drone_pos = self.state["position"].x_val
        y_drone_pos = self.state["position"].y_val
        z_drone_pos = self.state["position"].z_val
        camera_pose = airsim.Pose(
            airsim.Vector3r(x_drone_pos, y_drone_pos, z_drone_pos),
            airsim.to_quaternion(0, 0, yaw_drone_frame),
        )
        self.drone.simSetCameraPose(camera_type, camera_pose)
        # self.drone.simSetCameraPose("0", camera_pose)

        # Save the position of the drone to a csv file
        self._log_position_state(x_drone_pos, y_drone_pos, z_drone_pos)

        # Calculate energy consumption
        self.totalPower = 0
        try:
            for i in range(len(self.rotorData.rotors)):
                if i in {0, 1}:
                    power = -(
                        self.rotorData.rotors[i]["thrust"]
                        * self.rotorData.rotors[i]["torque_scaler"]
                        * self.rotorData.rotors[i]["speed"]
                    )
                else:
                    power = (
                        self.rotorData.rotors[i]["thrust"]
                        * self.rotorData.rotors[i]["torque_scaler"]
                        * self.rotorData.rotors[i]["speed"]
                    )
                self.totalPower += power
        except:
            pass
        
        print(f"Energy consumption : {round(self.totalPower, 2)}W")
        
        # Parse the FPV view and operate on it to get the bounding box + camera view parameters
        responses = self.drone.simGetImages(
            [
                airsim.ImageRequest(camera_type, airsim.ImageType.Scene, False, False),
                airsim.ImageRequest(camera_type, airsim.ImageType.DepthPlanar, True),
            ]
        )
        response = responses[0]
        rawImage = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        rawImage = rawImage.reshape(response.height, response.width, 3)
        (
            rawImage,
            xmin,
            ymin,
            xmax,
            ymax,
            width,
            height,
            detected,
            conf,
        ) = self.detectAndMark(rawImage)
        
        self.cam_coords["xmin"] = xmin
        self.cam_coords["ymin"] = ymin
        self.cam_coords["xmax"] = xmax
        self.cam_coords["ymax"] = ymax
        self.cam_coords["height"] = height
        self.cam_coords["width"] = width
        self.cam_coords["confidence"] = conf

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        # Depth Camera
        img_depth = np.asarray(responses[1].image_data_float)
        img_depth = img_depth.reshape(responses[1].height, responses[1].width)
        img_depth[img_depth > 16000] = np.nan
        img_depth = cv2.resize(img_depth, (1920, 1080), interpolation=cv2.INTER_AREA)
        img_depth_crop = img_depth[int(ymin) : int(ymax), int(xmin) : int(xmax)]
        
        try:
            self.depthDistance = int(np.nanmin(img_depth_crop))
            print(f"Distance is {self.depthDistance}")
        except:
            self.depthDistance = self.prev_depthDistance

        depth_range = np.array([np.nanmin(img_depth), np.nanmax(img_depth)])
        depth_map = np.around(
            (img_depth - depth_range[0]) * (255 - 0) / (depth_range[1] - depth_range[0])
        )

        try:
            (
                _,
                self.cam_coords["edge_x1"],
                self.cam_coords["edge_y1"],
                self.cam_coords["edge_x2"],
                self.cam_coords["edge_y2"],
            ) = self.edge_detection(depth_map)
        except:
            print("No lines detected")
            self.cam_coords["edge_x1"] = 0
            self.cam_coords["edge_y1"] = 0
            self.cam_coords["edge_x2"] = 0
            self.cam_coords["edge_y2"] = 0

        fake_return = np.zeros((84, 84, 1))

        return fake_return

    def _do_action(self, action):
        quad_offset, rotate = self.interpret_action(action)
        # quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityBodyFrameAsync(
            quad_offset[0],
            quad_offset[1],
            quad_offset[2],
            self.stepTime,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(True, rotate),
        )
        # Get torques from rotors DURING movement
        time.sleep(self.stepTime / 2)  # FIX ME, not an elegant solution
        self.rotorData = self.drone.getRotorStates()
        time.sleep(self.stepTime / 2)

    # Gradient (linear) reward for the bounding box location
    def reward_center(self, center, width, limit):
        if center >= 0 and center < (width / 2 - limit):
            reward = ((1 / (width / 2 - limit)) * center) - 1
        elif center >= (width / 2 - limit) and center <= (width / 2 + limit):
            reward = -(1 / limit) * abs(center - (width / 2)) + 1
        elif center > (width / 2 + limit) and center <= width:
            reward = -(1 / (width / 2 - limit)) * (center - (width / 2 + limit))
        else:
            reward = -1
        return reward

    def line_maximization(self, x1, y1, x2, y2, frame_width, frame_height):
        line_length = (
            (x2 - x1) ** 2 + (y2 - y1) ** 2
        ) ** 0.5  # Calculate the length of the line
        if x1 == x2:
            max_length = frame_height  # if vertical, max length is height
        elif y1 == y2:
            max_length = frame_width  # if horizontal, max length is width
        else:
            slope = (y2 - y1) / (x2 - x1)
            angle = math.atan(slope)
            max_length = (frame_width * math.cos(angle)) + (
                frame_height * math.sin(angle)
            )  # this, if at angle

        maximization_value = line_length / max_length

        return maximization_value

    def _compute_reward(self):
        """Calculate the reward based on the taken action

        Args:
            action (int): int mapping to the taken action

        Returns:
            reward: calculated reward
        """
        global curr_time, prev_time, detected, episode_length

        curr_time = perf_counter()
        reward = 0
        reward1 = 0
        reward2 = 0
        done = 0

        self.x_obj_middle = (
            self.cam_coords["xmin"]
            + (self.cam_coords["xmax"] - self.cam_coords["xmin"]) / 2
        )
        self.y_obj_middle = (
            self.cam_coords["ymin"]
            + (self.cam_coords["ymax"] - self.cam_coords["ymin"]) / 2
        )
        self.x_cam_middle = self.cam_coords["width"] / 2
        self.y_cam_middle = self.cam_coords["height"] / 2

        self.x_size = self.cam_coords["xmax"] - self.cam_coords["xmin"]
        self.y_size = self.cam_coords["ymax"] - self.cam_coords["ymin"]

        self.x_edge_middle = (
            self.edge_coords["edge_x1"]
            + (self.edge_coords["edge_x2"] - self.edge_coords["edge_x1"]) / 2
        )
        self.y_edge_middle = (
            self.edge_coords["edge_y1"]
            + (self.edge_coords["edge_y2"] - self.edge_coords["edge_y1"]) / 2
        )

        self.confidence = self.cam_coords["confidence"]

        if self.state["collision"]:
            reward = -100
            done = 1
            episode_length = 0
        else:
            if not detected:
                done = 1
                episode_length = 0
                print("Agent update - detection lost, exiting")

            print(f"Distance = {self.depthDistance}m")

            # REWARD 1
            delta_distance = self.depthDistance - self.prev_depthDistance
            if delta_distance < -1:
                print("Agent update - getting closer")
                reward1 += 1
                self.prev_depthDistance = self.depthDistance
            elif delta_distance > 1:
                print("Agent update - getting further")
                reward1 -= 1
                self.prev_depthDistance = self.depthDistance
            else:
                print("Agent update - stationiary")
                reward1 += 0
                self.prev_depthDistance = self.depthDistance

            reward_obj_center = self.reward_center(
                self.x_obj_middle, self.cam_coords["width"], 400
            ) + self.reward_center(self.y_obj_middle, self.cam_coords["height"], 400)
            reward1 += reward_obj_center

            # REWARD 1
            reward_edge_center = self.reward_center(
                self.x_edge_middle, self.cam_coords["width"], 400
            ) + self.reward_center(self.y_edge_middle, self.cam_coords["height"], 400)
            reward2 += reward_edge_center
            reward2 += self.line_maximization(
                self.edge_coords[
                    "edge_x1"
                ],  # how big is the line wrt to the camera view
                self.edge_coords["edge_y1"],  # outputs 0-1 range
                self.edge_coords["edge_x2"],
                self.edge_coords["edge_y2"],
                self.cam_coords["width"],
                self.cam_coords["height"],
            )

            # Weighted sum of the two rewards (dynamic reward function)
            W1 = 0.5 * (
                1 + np.tanh(2 * (self.depthDistance - 100))
            )  # Smooth transition around the 30m mark
            W2 = 1 - W1
            reward = reward + W1 * reward1 + W2 * reward2

            # global prev_E_time, prev_E
            # variables = self.generate_energy_consumption_variables(prev_E_time)
            # reward_energy = self.calculate_energy_consumption_reward(prev_E, *variables)
            # print(f"Energy consumption reward: {reward_energy}")
            # reward += reward_energy
            energy_reward = self.calculate_torque_energy_reward(self.totalPower)
            print(f"Energy consumption reward: {round(energy_reward, 2)}")
            reward += energy_reward
            
            fog_conf_reward = self.calculate_fog_conf_reward(self.fog_level, self.cam_coords["confidence"])
            print(f"Fog/Confidence reward: {fog_conf_reward}")
            reward += fog_conf_reward
            

            if episode_length >= 200 or self.depthDistance < 50.0:
                print(
                    "Agent stopped - max time_step in episode exceeded or distance < 50m"
                )
                done = 1
                episode_length = 0

        return reward, done

    def step(self, action):
        global episode_length
        self._do_action(action)
        obs = self._get_obs()
        episode_length += 1
        print("Episode - timestep: ", episode_length)
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        global prev_E
        prev_E = 1
        self._setup_flight()
        return self._get_obs()

    def interpret_action(self, action):
        rotate = 0
        quad_offset = (0, 0, 0)

        if action == 0:  # FRONT
            quad_offset = (self.step_length, 0, 0)
            rotate = 0
        elif action == 1:  # RIGHT
            quad_offset = (0, 0, 0)
            rotate = 2
        elif action == 2:  # LEFT
            quad_offset = (0, 0, 0)
            rotate = -2
        elif action == 3:  # UP
            quad_offset = (0, 0, self.step_length)
            rotate = 0
        elif action == 4:  # DOWN
            quad_offset = (0, 0, -self.step_length)
            rotate = 0
        else:  # STOP
            quad_offset = (0, 0, 0)
            rotate = 0

        return quad_offset, rotate

    def _log_position_state(self, position_x: int, position_y: int, position_z: int):
        """Save position of the drone into a CSV file

        Args:
            position_x (int): Position in X in world coordinates
            position_y (int): Position in Y in world coordinates
            position_z (int): Position in Z in world coordinates
        """
        with open("drone_position.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            # writer.writerow(["X", "Y", "Z"])
            writer.writerow([position_x, position_y, position_z])

    def calculate_energy_consumption_reward(
        self,
        prev_E: float,
        hovering: float,
        horizontal: float,
        vertical_up: float,
        vertical_down: float,
        altitude: float,
        payload: int,
    ) -> float:
        """Calculate energy consumption reward.

        Args:
            prev_E (float): Previous energy consumption reward
            hovering (float): Total hovering time
            horizontal (float): Total horizontal flying time
            vertical_up (float): Total vertical flying upwards distance
            vertical_down (float): Total vertical flying downwards distance
            altitude (float): Relative altitude of hovering
            payload (int): Payload weight (grams)

        Returns:
            float: energy consumption reward

        """
        # E = Idle mode + armed mode + Take off + flying vertically upward +
        #    hovering + payload + flying horizontally + flying vertically downward

        # R consumption equation:
        curr_E = (
            -278.695 + 8.195 * 0 + 29.027 * 0 - 0.432 * pow(0,2)
            + 3.786 * 0
            + 315 * vertical_up
            + (4.917 * altitude + 275.204) * hovering
            + (0.311 * payload + 1.735) * horizontal
            + 308.709 * horizontal
            + 68.956 * vertical_down
        )

        # Calculate the change in energy
        delta_E = curr_E - prev_E
        # Calculate the reward value
        if delta_E == 0:
            E_new = 1  # No energy was used, so the reward is maximum
        else:
            slope = -1 / prev_E
            E_new = slope * abs(delta_E) + 1
        print(f"ACTUAL REWARD = {E_new}")
        # Ensure that the reward value is between -1 and 1
        E_new = max(-1, min(E_new, 1))
        prev_E == curr_E
        return E_new

    def generate_energy_consumption_variables(self, prev_E_time_t):
        hovering_t = 0
        horizontal_t = 0
        vertical_up_t = 0
        vertical_down_t = 0
        altitude_t = 0
        payload_t = 1000
        # Calculate elapsed time since last call to reward function
        current_time = time.time()
        elapsed_time = current_time - prev_E_time_t

        # Calculate total hovering time
        if round(self.state["velocity"].z_val) == 0:
            hovering_t += elapsed_time

        # Calculate total horizontal flying time
        horizontal_t += (
            elapsed_time
            if self.state["velocity"].x_val != 0 or self.state["velocity"].y_val != 0
            else 0
        )

        # Calculate total vertical flying upwards and downwards distance
        if -self.state["velocity"].z_val > 0:
            vertical_up_t += -self.state["velocity"].z_val * elapsed_time
        elif -self.state["velocity"].z_val < 0:
            vertical_down_t += abs(self.state["velocity"].z_val) * elapsed_time

        # Calculate relative altitude of hovering
        altitude_t = -self.state["position"].z_val
        
        
        data = [    ['Hovering time', hovering_t],
            ['Horizontal flying time', horizontal_t],
            ['Vertical flying upwards distance', vertical_up_t],
            ['Vertical flying downwards distance', vertical_down_t],
            ['Altitude of hovering', altitude_t],
            ['Payload weight', payload_t]
        ]
        from tabulate import tabulate
        table = tabulate(data, headers=['Parameter', 'Value'])
        print(table)
        return (
            hovering_t,
            horizontal_t,
            vertical_up_t,
            vertical_down_t,
            altitude_t,
            payload_t,
        )

    def calculate_torque_energy_reward(self, total_energy):
        """Linearly maps consumned energy from (0-400)[W] to (1,-1)

        Args:
            total_energy (float): Consumned energy in Watts

        Returns:
            float: Reward (-1, 1)
        """
        input_min = 100
        input_max = 200
        output_min = 1
        output_max = -1

        # Perform the linear mapping
        energy_reward = ((total_energy - input_min) / (input_max - input_min)) * (
            output_max - output_min
        ) + output_min
        
        if energy_reward > 1:
            energy_reward = 1
        elif energy_reward < -1:
            energy_reward = -1
        else:
            energy_reward = energy_reward

        return energy_reward
    def calculate_fog_conf_reward(self, x_fog, y_conf):
        """Calculated the reward based on the visibility level and confidence score.

        Args:
            x_fog (float): Fog Level
            y_conf (float): Object Detection Confidence

        Returns:
            float: reward
        """        
        if x_fog > 1:
            x_fog = 1
        elif x_fog < 0:
            x_fog = 0
        if y_conf > 1:
            y_conf = 1
        elif y_conf < 0:
            y_conf = 0

        if x_fog + y_conf >= 1:
            return float(x_fog + y_conf - 1)
        elif (x_fog + y_conf < 1) and ((-7/5 * x_fog + 0.7) > y_conf):
                return float(2 * x_fog + (0.5/0.35) * y_conf - 1)
        elif (x_fog + y_conf < 1) and ((-7/5 * x_fog + 0.7) < y_conf):
            return float(0.0)
        else:
            return float(-1.0)
