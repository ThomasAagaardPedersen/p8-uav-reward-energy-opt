from pathlib import Path

from lib.extract_csv_tensor import extract_data_from_tensorboard
from lib.plot_csv import plot_csv_logs

plot_title = r"\text{Torque-based energy consumption with continuation after lost detection}"
log_folder = "Torque_Reward_and_No_lost_detection_test_1_1683801433.881646_1"

# Usage example
rollout = [
    "rollout/ep_len_mean",
    "rollout/ep_rew_mean",
    "rollout/exploration_rate",
    "train/loss",
]

log_dir = Path(__file__).parent.parent / log_folder
output_dir = Path(__file__).parent.parent / log_folder


args = extract_data_from_tensorboard(log_dir, rollout)
plot_csv_logs(*args, plot_title)
