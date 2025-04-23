# Defining valuables
"""
t_1 = 0
t_2 = 0
V = 0
D = 0
H = 0
t_3 = 0
L = 0
t_4 = 0
D_1 = 0
prev_E = 0
"""


def calculate_reward(prev_E, hovering, horizontal, vertical_up, vertical_down, payload):
    """"""
    #E = Idle mode + armed mode + Take off + flying vertically upward +
    #    hovering + payload + flying horizontally + flying vertically downward
    

    # R consumption equation:
    curr_E = (
        -278.695 + 8.195 * t_1 + 29.027 * t_2 - 0.432 * V
        ^ 2
        + 3.786 * V
        + 315 * D
        + (4.917 * H + 275.204) * t_3
        + (0.311 * L + 1.735) * t_3
        + 308.709 * t_4
        + 68.956 * D_1
    )

    # Calculate the change in energy
    delta_E = curr_E - prev_E
    # Calculate the reward value
    if delta_E == 0:
        E_new = 1  # No energy was used, so the reward is maximum
    else:
        slope = -1 / prev_E
        E_new = slope * abs(delta_E) + 1

    # Ensure that the reward value is between -1 and 1
    E_new = max(-1, min(E_new, 1))
    prev_E == curr_E
    return E_new


"""
The function first calculates the change in energy delta_E by subtracting the previous energy value prev_E from the current energy value curr_E.

If no energy was used (delta_E == 0), the reward value E_new is set to maximum (1).

If energy was used or saved (delta_E != 0), the reward value E_new is calculated based on the linear equation y = (-1 / prev_E) * x + 1. We calculate the absolute value of delta_E to ensure that the reward value increases as the magnitude of delta_E increases.

To calculate the slope of the line m, we use the formula m = (y2 - y1) / (x2 - x1), where (x1, y1) = (0, 1) and (x2, y2) = (prev_E, 0). So the slope is calculated as:

scss
Copy code
m = (0 - 1) / (prev_E - 0) = -1 / prev_E
The equation for the line is then y = (-1 / prev_E) * x + 1. We substitute x = abs(delta_E) into this equation to get the reward value y.

To ensure that the reward value is between -1 and 1, we use the max and min functions to clamp the value of E_new within that range.

I hope this updated code meets your requirements.
"""
