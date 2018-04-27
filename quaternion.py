import numpy as np


def add(qa, qb):
    s = qa[0, 0] * qb[0, 0] - qa[1, 0] * qb[1, 0] - qa[2, 0] * qb[2, 0] - qa[3, 0] * qb[3, 0]
    x = qa[0, 0] * qb[1, 0] + qa[1, 0] * qb[0, 0] + qa[2, 0] * qb[3, 0] - qa[3, 0] * qb[2, 0]
    y = qa[0, 0] * qb[2, 0] - qa[1, 0] * qb[3, 0] + qa[2, 0] * qb[0, 0] + qa[3, 0] * qb[1, 0]
    z = qa[0, 0] * qb[3, 0] + qa[1, 0] * qb[2, 0] - qa[2, 0] * qb[1, 0] + qa[3, 0] * qb[0, 0]

    q = np.array([[s], [x], [y], [z]])

    return q


def conj(q):
    s = q[:1, :]
    v = q[1:, :]

    q_conj = np.concatenate((s, -v), axis=0)

    return q_conj


def norm(q):
    return np.sqrt(np.dot(q.T, q))


def inv(q):
    return conj(q) / (norm(q) * norm(q))


def subtract(qa, qb):
    qb_inv = inv(qb)

    return add(qa, qb_inv)


def vec2quat(vec):
    if np.linalg.norm(vec) < 10e-5:
        quat = np.array([[1], [0], [0], [0]])
        return quat

    angle = np.linalg.norm(vec)
    axis = vec / angle

    cos_term = np.array([[np.cos(angle / 2)]])

    quat = np.concatenate((cos_term, axis * np.sin(angle / 2)), axis=0)

    return quat


def quat2vec(quat):
    angle = 2 * np.arccos(quat[:1, :])
    vec = quat[1:, :] / np.sin(angle / 2)

    return angle * vec


def quat2rot(quat):
    quat = quat / np.linalg.norm(quat)
    q0 = quat[0, 0]
    q1 = quat[1, 0]
    q2 = quat[2, 0]
    q3 = quat[3, 0]

    R = np.array([[(q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3), (2 * (q1 * q2 - q0 * q3)), (2 * (q1 * q3 + q0 * q2))],
                  [(2 * (q1 * q2 + q0 * q3)), (q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3), (2 * (q2 * q3 - q0 * q1))],
                  [(2 * (q1 * q3 - q0 * q2)), (2 * (q2 * q3 + q0 * q1)), (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)]])

    return R


def average_quaternion(quats):
    D, N = quats.shape

    # avg_quat_cov = np.dot(quats, quats.T) / N
    #
    # eig, eig_vec = np.linalg.eig(np.nan_to_num(avg_quat_cov))
    # max_eig_index = np.argmax(eig)
    # avg_quat = eig_vec[:, max_eig_index, np.newaxis]

    # Maximum number of iterations
    num_iter = 1000

    # Initialize mean quaternion and sum of error vectors
    avg_quat = np.ones((4, 1)) * 0.5
    e_vec_sum = np.zeros((3, 1))

    # Gradient descent introduced in paper
    for i in range(num_iter):
        for k in range(N):
            e_i = subtract(quats[:, k, np.newaxis], avg_quat)
            e_vec_sum = e_vec_sum + quat2vec(e_i)

        e_vec_avg = e_vec_sum / N

        # Break if converges
        if np.linalg.norm(e_vec_avg) < 10e-5:
            break

        e_vec_sum = np.zeros((3, 1))
        e = vec2quat(e_vec_avg)

        avg_quat = add(e, avg_quat)

    return avg_quat
