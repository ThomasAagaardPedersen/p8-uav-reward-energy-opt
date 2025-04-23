import airsim
import time
from bat_optimization import calculate_reward

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

# Take off and hover for a few seconds
client.takeoffAsync().join()
time.sleep(5)

# Get drone's initial position
pose = client.simGetVehiclePose()

# Store initial altitude
initial_altitude = pose.position.z_val

# Initialize variables for tracking flight data
total_hover_time = 0
total_horizontal_time = 0
total_vertical_up_distance = 0
total_vertical_down_distance = 0
payload = 0

# Begin flight loop
while True:
    # Get current drone position and velocity
    pose = client.simGetVehiclePose()
    vel = client.simGetGroundTruthKinematics().linear_velocity

    # Get current altitude relative to initial altitude
    altitude = pose.position.z_val - initial_altitude

    # Check if drone is hovering or flying horizontally
    if round(vel.x_val) == 0 and round(vel.y_val) == 0:
        # Drone is hovering
        total_hover_time += 1
    else:
        # Drone is flying horizontally
        total_horizontal_time += 1

    # Check if drone is flying upwards or downwards
    if vel.z_val > 0:
        # Drone is flying upwards
        total_vertical_up_distance += vel.z_val
    elif vel.z_val < 0:
        # Drone is flying downwards
        total_vertical_down_distance += abs(vel.z_val)

    # Get payload weight (in kg)
    # Replace this with your own code to read the drone's payload weight
    payload = 0

    # Break out of the loop if the drone has landed
    if altitude <= 0:
        break

    # Wait for a short time before getting the next set of data
    time.sleep(0.1)

# Print flight data
print("Total hovering time: {} seconds".format(total_hover_time))
print("Total horizontal flying time: {} seconds".format(total_horizontal_time))
print(
    "Total vertical flying upwards distance: {} meters".format(
        total_vertical_up_distance
    )
)
print(
    "Total vertical flying downwards distance: {} meters".format(
        total_vertical_down_distance
    )
)
print("Payload weight: {} kg".format(payload))
