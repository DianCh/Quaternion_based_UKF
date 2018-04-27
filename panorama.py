import numpy as np


def initialize_canvas(H, W):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    ratio = H / np.pi
    return canvas, ratio


def stitch_one_img(img, Rot, canvas, ratio):

    # Generate XYZ mesh of a spherical image
    X, Y, Z = generate_mesh(img.shape)

    # Transform the pixels to given orientation
    X_prime, Y_prime, Z_prime = transform_pixels(X, Y, Z, Rot)

    # Convert back the cartesian coordinates to spherical
    phi_Y, theta_X = cartesian_to_spherical(X_prime, Y_prime, Z_prime)
    # Shift theta to range [0, 2*pi]
    theta_X = theta_X + np.pi

    # Transform the angles to pixel coordinates
    X_final = np.array(theta_X * ratio, dtype=np.int32)
    Y_final = np.array(phi_Y * ratio, dtype=np.int32)

    # Copy the pixel values onto canvas
    canvas = copy_pixels(image=img, canvas=canvas, X=X_final, Y=Y_final)

    return canvas


def generate_mesh(shape, FOV_H=np.pi/4, FOV_W=np.pi/3):
    # Extract size information
    H, W, _ = shape
    center_H = H / 2.0
    center_W = W / 2.0

    x = np.arange(0, W, 1)
    y = np.arange(0, H, 1)
    xx, yy = np.meshgrid(x, y, sparse=False)
    xx = xx - center_W
    yy = yy - center_H

    # Angle covered per pixel
    theta_unit = FOV_W / W
    phi_unit = FOV_H / H

    theta = xx * theta_unit
    phi = yy * phi_unit + np.pi / 2

    # Compute the spherical coordinates of each pixel
    X, Y, Z = spherical_to_cartesian(theta, phi)

    return X, Y, Z


def spherical_to_cartesian(theta, phi, rho=1):
    X = rho * np.sin(phi) * np.cos(theta)
    Y = rho * np.sin(phi) * np.sin(theta)
    Z = rho * np.cos(phi)
    return X, Y, Z


def cartesian_to_spherical(X, Y, Z):
    rho = np.sqrt(X * X + Y * Y + Z * Z)
    phi = np.arccos(Z / rho)
    theta = np.arctan2(Y, X)
    return phi, theta


def transform_pixels(X, Y, Z, Rot):
    # Extract size information
    H, W = X.shape

    # Construct all columns for coordinates
    coords = np.zeros((3, H, W))
    coords[0, :, :] = X
    coords[1, :, :] = Y
    coords[2, :, :] = Z
    coords_reshaped = coords.reshape((3, H * W))

    # Transform the pixels
    coords_reshaped = np.dot(Rot, coords_reshaped)

    # Reshape back
    X_prime = coords_reshaped[0, :]
    Y_prime = coords_reshaped[1, :]
    Z_prime = coords_reshaped[2, :]

    return X_prime, Y_prime, Z_prime


def copy_pixels(image, canvas, X, Y):
    # Extract size information
    H, W, _ = image.shape

    # Flip the image horizontally, viewing it from inside the sphere
    image = image[:, ::-1, :]

    # Construct coordinates for indexing
    rows = np.array(Y.reshape(H * W), dtype=np.int32)
    columns = np.array(X.reshape(H * W), dtype=np.int32)

    channel_1 = np.zeros(H * W, dtype=np.int32)
    channel_2 = np.ones(H * W, dtype=np.int32)
    channel_3 = np.ones(H * W, dtype=np.int32) * 2

    image_channel_1 = image[:, :, 0].reshape(H * W)
    image_channel_2 = image[:, :, 1].reshape(H * W)
    image_channel_3 = image[:, :, 2].reshape(H * W)

    canvas[rows, columns, channel_1] = image_channel_1
    canvas[rows, columns, channel_2] = image_channel_2
    canvas[rows, columns, channel_3] = image_channel_3

    return canvas
