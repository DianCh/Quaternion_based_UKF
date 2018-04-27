from scipy import io
import numpy as np


def load_omega_acc(number, chop=550):
    # Get the corresponding raw data
    filename = "imu/imuRaw" + str(number) + ".mat"
    imuRaw = io.loadmat(filename)

    # Extract data
    data = imuRaw["vals"]

    # Remove the biases
    biases = np.mean(data[:, 1:300], axis=1, keepdims=True)
    data = data - biases

    # Extract acceleration
    acc_data = data[0:3, :-chop] + np.array([[0], [0], [93]])

    # Extract angular velocity
    omega_data_z = data[3:4, :-chop]
    omega_data_xy = data[4:6, :-chop]
    omega_data = np.concatenate((omega_data_xy, omega_data_z), axis=0)
    # omega_data = omega_data - np.array([[374], [375], [370]])

    # Convert to physical units
    sensitivity_acc = 0.3                        # 300mV typical
    scale_acc = 3.3 / sensitivity_acc / 1023     # Vref = 3.3V typical
    acc_data = acc_data * scale_acc

    sensitivity_omega = 3.33 / 1000                  # 0.83mV / 3.33mV typical
    scale_omega = 3.3 / sensitivity_omega / 1023     # Vref = 3.3
    omega_data = omega_data * scale_omega * np.pi / 180

    # Extract time stamps
    imu_time_stamps = imuRaw["ts"][:, :-chop]

    return omega_data, acc_data, imu_time_stamps


def load_vicon(number, chop=550):
    # Get the corresponding raw data
    filename = "vicon/viconRot" + str(number) + ".mat"
    vicon_raw = io.loadmat(filename)

    # Extract rotation matrices and time stamps
    vicon_rots = vicon_raw["rots"][:, :, :-chop]
    vicon_time_stamps = vicon_raw["ts"][:, :-chop]

    return vicon_rots, vicon_time_stamps


def load_camera_image(number, chop=1):
    # Get the corresponding raw data
    filename = "cam/cam" + str(number) + ".mat"
    cam_raw = io.loadmat(filename)

    # Extract images and time stamps
    cam_imgs = cam_raw["cam"][:, :, :, :-chop]
    cam_time_stamps = cam_raw["ts"][:, :-chop]

    return cam_imgs, cam_time_stamps
