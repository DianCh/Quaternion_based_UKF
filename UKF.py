import quaternion as qtn
import scipy
import numpy as np
import matplotlib.pyplot as plt


def initialize_mean_cov():
    x0 = np.array([[1], [0], [0], [0]])

    # P0 = 10 * gen_spd(3)
    P0 = np.eye(3) * 100

    return x0, P0


def initialize_noises():
    # Q = 10 * gen_spd(3)
    # R = 10 * gen_spd(3)
    Q = np.diag(np.ones(3) * 100)
    R = np.diag(np.ones(3) * 100)

    return Q, R


def sequential_filtering(omega_data, z_data, time_stamps):
    # Initialize belief
    x0, P0 = initialize_mean_cov()

    # Initialize noises
    Q, R = initialize_noises()

    # Count the number of effective data
    k = omega_data.shape[1]  # Should be the same as z_data.shape[1]

    # Initialize a series of Rotation matrices
    Rot = np.zeros((3, 3, k-1))     # Get rid of the first frame

    x_prev, P_prev = x0, P0
    for i in range(1, k):
        delta_t = time_stamps[0, i] - time_stamps[0, i-1]
        x, P = one_step_filtering(x_prev=x_prev,
                                  P_prev=P_prev,
                                  input_omega=omega_data[:, i, np.newaxis],
                                  delta_t=delta_t,
                                  motion_noise=Q,
                                  z=z_data[:, i, np.newaxis],
                                  measure_noise=R)
        x_prev = x
        P_prev = P
        Rot[:, :, i-1] = qtn.quat2rot(x)

    return Rot, time_stamps[:, 1:]


def one_step_filtering(x_prev, P_prev, input_omega, delta_t, motion_noise, z, measure_noise):
    n = x_prev.shape[0] - 1
    N = 2 * n
    gravity = 1

    # Motion update
    avg_quat, Y_sigma = motion_update(x_prev, P_prev, motion_noise, input_omega, delta_t)

    # Measurement update
    avg_z, Z_sigma = measurement_update(Y_sigma, gravity)

    # Compute innovation
    # Equation (44)
    innovation = z - avg_z

    # Compute covariances
    # Equation (63)
    q_W_prime = sigma_remove_center(Y_sigma, avg_quat)
    W_prime = sigma_noise_back(q_W_prime)

    # Equation (64)
    P_pred = np.dot(W_prime, W_prime.T) / N

    # Equation (68)
    P_zz = np.dot((Z_sigma - avg_z), (Z_sigma - avg_z).T) / N

    # Equation (69)
    P_vv = P_zz + measure_noise

    # Equation (70)
    P_xz = np.dot(W_prime, (Z_sigma - avg_z).T) / N

    # Compute Kalman gain
    # Equation (72)
    K = np.dot(P_xz, np.linalg.inv(P_vv))

    # Compute final a-posteriori belief
    x = qtn.add(avg_quat, qtn.vec2quat(np.dot(K, innovation)))
    P = P_pred - np.dot(np.dot(K, P_vv), K.T)

    return x, P


def sequential_movements(omega_data, time_stamps):
    # Initialize belief
    x0, P0 = initialize_mean_cov()

    # Initialize noises
    Q, R = initialize_noises()

    # Count the number of effective data
    k = omega_data.shape[1]  # Should be the same as z_data.shape[1]

    # Initialize a series of Rotation matrices
    Rot = np.zeros((3, 3, k-1))     # Get rid of the first frame

    x_prev, P_prev = x0, P0
    for i in range(1, k):
        delta_t = time_stamps[0, i] - time_stamps[0, i-1]
        x, P = one_step_movement(x_prev=x_prev,
                                 P_prev=P_prev,
                                 input_omega=omega_data[:, i, np.newaxis],
                                 delta_t=delta_t,
                                 motion_noise=Q)
        x_prev = x
        P_prev = P
        Rot[:, :, i-1] = qtn.quat2rot(x)

    return Rot, time_stamps[:, 1:]


def one_step_movement(x_prev, P_prev, input_omega, delta_t, motion_noise):
    n = x_prev.shape[0] - 1
    N = 2 * n
    gravity = 1

    # Motion update
    avg_quat, Y_sigma = motion_update(x_prev, P_prev, motion_noise, input_omega, delta_t)

    # Compute covariances
    # Equation (63)
    q_W_prime = sigma_remove_center(Y_sigma, avg_quat)
    W_prime = sigma_noise_back(q_W_prime)

    # Equation (64)
    P_pred = np.dot(W_prime, W_prime.T) / N

    return avg_quat, P_pred


def low_pass_filtering(acc_data, alpha=0.7):
    D, N = acc_data.shape

    acc_data_filtered = np.zeros((D, 0))
    acc_data_filtered = np.concatenate((acc_data_filtered, acc_data[:, 0, np.newaxis]), axis=1)

    for i in range(1, N):
        filtered = alpha * acc_data[:, i] + (1 - alpha) * acc_data_filtered[:, i-1]
        acc_data_filtered = np.concatenate((acc_data_filtered, filtered[:, np.newaxis]), axis=1)

    return acc_data_filtered[:, 1:]


def pitch_roll_from_acc(acc_data):

    pitch = np.arctan2(-acc_data[1, :], acc_data[2, :])
    roll = np.arctan2(acc_data[0, :],
                      np.sqrt(acc_data[1, :] * acc_data[1, :] + acc_data[2, :] * acc_data[2, :]))

    return pitch, roll


def motion_update(x_prev, P_prev, motion_noise, input_omega, delta_t):
    n = P_prev.shape[0]

    # Equation (35) and (36)
    P_noisy = P_prev + motion_noise

    S = scipy.linalg.cholesky(P_noisy) * np.sqrt(2 * n)
    W = np.concatenate((S, -S), axis=1)

    # Equation (14) - (16)
    q_W = sigma_noise(W)

    # Equation (34)
    X_sigma = sigma_move_center(x_prev, q_W)

    # Equation (37) and implicitly (17)
    Y_sigma = motion_model(X_sigma, input_omega, delta_t)

    # Method Prof. Lee introduced in class
    avg_quat = qtn.average_quaternion(Y_sigma)

    return avg_quat, Y_sigma


def measurement_update(Y_sigma, gravity):
    # Equation (40)
    Z_sigma = measure_model(Y_sigma, gravity)

    # Equation (41)
    avg_z = np.mean(Z_sigma, axis=1, keepdims=True)

    return avg_z, Z_sigma


def gen_spd(D):
    # generate random sigma using U'XU
    d = np.random.rand(D)
    X = np.diag(d)
    U = scipy.linalg.orth(np.random.rand(D, D))
    sigma = np.dot(np.dot(U.T, X), U)

    return sigma


def sigma_noise(W):
    N = W.shape[1]
    q_W = np.zeros((4, 0))

    for i in range(N):
        quat = qtn.vec2quat(W[:, i, np.newaxis])
        q_W = np.concatenate((q_W, quat), axis=1)

    return q_W


def sigma_noise_back(q_W_prime):
    N = q_W_prime.shape[1]
    W_prime = np.zeros((3, 0))

    for i in range(N):
        vec = qtn.quat2vec(q_W_prime[:, i, np.newaxis])
        W_prime = np.concatenate((W_prime, vec), axis=1)

    return W_prime


def sigma_move_center(x_prev, q_W):
    N = q_W.shape[1]
    X_sigma = np.zeros((4, 0))

    for i in range(N):
        quat = qtn.add(x_prev, q_W[:, i, np.newaxis])
        X_sigma = np.concatenate((X_sigma, quat), axis=1)

    return X_sigma


def sigma_remove_center(Y_sigma, x_pred):
    N = Y_sigma.shape[1]
    q_W_prime = np.zeros((4, 0))

    for i in range(N):
        quat = qtn.subtract(Y_sigma[:, i, np.newaxis], x_pred)
        q_W_prime = np.concatenate((q_W_prime, quat), axis=1)

    return q_W_prime


def motion_model(X_sigma, omega, delta_t):
    N = X_sigma.shape[1]
    Y_sigma = np.zeros((4, 0))

    rotation = omega * delta_t
    q_rot = qtn.vec2quat(rotation)

    for i in range(N):
        quat = qtn.add(X_sigma[:, i, np.newaxis], q_rot)
        Y_sigma = np.concatenate((Y_sigma, quat), axis=1)

    return Y_sigma


def measure_model(Y_sigma, gravity):
    N = Y_sigma.shape[1]
    Z_sigma = np.zeros((3, 0))

    for i in range(N):
        R = qtn.quat2rot(Y_sigma[:, i, np.newaxis]).T
        z_pred = R[:, 2, np.newaxis] * gravity
        Z_sigma = np.concatenate((Z_sigma, z_pred), axis=1)

    return Z_sigma


def rotation_to_Euler(Rots):

    N = Rots.shape[2]
    Eulers = np.zeros((3, 0))

    for i in range(N):
        R = Rots[:, :, i]

        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        euler = np.array([[x], [y], [z]])
        Eulers = np.concatenate((Eulers, euler), axis=1)

    return Eulers


def plot_with_vicon(Rots_est, time_stamps_est, Rots_gyro, roll, pitch, time_stamps_imu, vicon_rots, vicon_time_stamps,
                    data_number, show_figure=False, comments="(with vicon)"):
    # Convert the rotation matrices to Euler angles
    Eulers_est = rotation_to_Euler(Rots_est)
    Eulers_gyro = rotation_to_Euler(Rots_gyro)
    Eulers_vicon = rotation_to_Euler(vicon_rots)

    # Unravel the time axis
    x_est = time_stamps_est[0, :] - time_stamps_est[0, 0]
    x_gyro = time_stamps_imu[0, :] - time_stamps_est[0, 0]
    x_vicon = vicon_time_stamps[0, :] - time_stamps_est[0, 0]

    # Extract the component to plot
    y1_est = Eulers_est[0, :]
    y1_gyro = Eulers_gyro[0, :]
    y1_acc = roll
    y1_vicon = Eulers_vicon[0, :]

    y2_est = Eulers_est[1, :]
    y2_gyro = Eulers_gyro[1, :]
    y2_acc = pitch
    y2_vicon = Eulers_vicon[1, :]

    y3_est = Eulers_est[2, :]
    y3_gyro = Eulers_gyro[2, :]
    y3_vicon = Eulers_vicon[2, :]

    # Start plotting
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(x_est, y1_est, 'b-', label="UKF estimation")
    plt.plot(x_gyro, y1_gyro, 'c-', label="gyro estimation")
    plt.plot(x_gyro, y1_acc, 'm-', label="acc estimation")
    plt.plot(x_vicon, y1_vicon, 'r-', label="Vicon ground truth")
    title = 'estimations vs. ground truth-' + 'dataset ' + str(data_number) + comments
    plt.title(title)
    plt.ylabel('roll')
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 2)
    plt.plot(x_est, y2_est, 'b-')
    plt.plot(x_gyro, y2_gyro, 'c-')
    plt.plot(x_gyro, y2_acc, 'm-')
    plt.plot(x_vicon, y2_vicon, 'r-')
    plt.ylabel('pitch')

    plt.subplot(3, 1, 3)
    plt.plot(x_est, y3_est, 'b-')
    plt.plot(x_gyro, y3_gyro, 'c-')
    plt.plot(x_vicon, y3_vicon, 'r-')
    plt.ylabel('yaw')
    plt.xlabel('time')

    filename = "Euler_results_" + str(data_number) + comments
    plt.savefig(filename)

    if show_figure:
        plt.show()

    plt.close()

    return 0


def plot_without_vicon(Rots_est, time_stamps_est, Rots_gyro, roll, pitch, time_stamps_imu,
                       data_number, show_figure=False, comments="(without vicon)"):
    # Convert the rotation matrices to Euler angles
    Eulers_est = rotation_to_Euler(Rots_est)
    Eulers_gyro = rotation_to_Euler(Rots_gyro)

    # Unravel the time axis
    x_est = time_stamps_est[0, :] - time_stamps_est[0, 0]
    x_gyro = time_stamps_imu[0, :] - time_stamps_est[0, 0]

    # Extract the component to plot
    y1_est = Eulers_est[0, :]
    y1_gyro = Eulers_gyro[0, :]
    y1_acc = roll

    y2_est = Eulers_est[1, :]
    y2_gyro = Eulers_gyro[1, :]
    y2_acc = pitch

    y3_est = Eulers_est[2, :]
    y3_gyro = Eulers_gyro[2, :]

    # Start plotting
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(x_est, y1_est, 'b-', label="UKF estimation")
    plt.plot(x_gyro, y1_gyro, 'c-', label="gyro estimation")
    plt.plot(x_gyro, y1_acc, 'm-', label="acc estimation")
    title = 'estimations only-' + 'dataset ' + str(data_number) + comments
    plt.title(title)
    plt.ylabel('roll')
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 2)
    plt.plot(x_est, y2_est, 'b-')
    plt.plot(x_gyro, y2_gyro, 'c-')
    plt.plot(x_gyro, y2_acc, 'm-')
    plt.ylabel('pitch')

    plt.subplot(3, 1, 3)
    plt.plot(x_est, y3_est, 'b-')
    plt.plot(x_gyro, y3_gyro, 'c-')
    plt.ylabel('yaw')
    plt.xlabel('time')

    filename = "Euler_results_" + str(data_number) + comments
    plt.savefig(filename)

    if show_figure:
        plt.show()

    plt.close()

    return 0
