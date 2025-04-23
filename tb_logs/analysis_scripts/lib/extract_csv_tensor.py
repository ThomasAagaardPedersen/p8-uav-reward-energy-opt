import csv
import os

from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator


def extract_csv_from_tensorboard(log_dir, output_dir, rollout):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get all available tags
    tags = event_acc.Tags()

    for tag in tags["scalars"]:
        if tag in rollout:
            csv_file_path = os.path.join(
                output_dir, str(tag).replace("/", "\\") + ".csv"
            )

        # Open CSV file for writing
        with open(csv_file_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)

            # Write CSV header
            writer.writerow(["step", "wall_time", "value"])

            # Extract data for each scalar event and write to CSV
            for scalar_event in event_acc.Scalars(tag):
                writer.writerow(
                    [scalar_event.step, scalar_event.wall_time, scalar_event.value]
                )

        print(f"CSV file saved for tag '{tag}': {csv_file_path}")


def extract_data_from_tensorboard(log_dir, rollout):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get all available tags
    tags = event_acc.Tags()
    data_rew = []
    data_len = []
    data_exp = []
    data_loss = []

    for tag in tags["scalars"]:
        if tag in rollout:
            if tag == rollout[0]:
                for scalar_event in event_acc.Scalars(tag):
                    data_rew.append(
                        [scalar_event.step, scalar_event.wall_time, scalar_event.value]
                    )
            if tag == rollout[1]:
                for scalar_event in event_acc.Scalars(tag):
                    data_len.append(
                        [scalar_event.step, scalar_event.wall_time, scalar_event.value]
                    )
            if tag == rollout[2]:
                for scalar_event in event_acc.Scalars(tag):
                    data_exp.append(
                        [scalar_event.step, scalar_event.wall_time, scalar_event.value]
                    )
            if tag == rollout[3]:
                for scalar_event in event_acc.Scalars(tag):
                    data_loss.append(
                        [scalar_event.step, scalar_event.wall_time, scalar_event.value]
                    )

    return data_rew, data_len, data_exp, data_loss
