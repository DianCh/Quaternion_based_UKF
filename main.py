import numpy as np
import matplotlib.pyplot as plt
import cv2

import load_data as ld
import UKF
import panorama as pano


def main(data_number, generate_video=False, have_vicon=True):
    # Define the size of output panorama
    H, W = (540, 1080)
    canvas, ratio = pano.initialize_canvas(540, 1080)

    # Load IMU data
    omega_data, acc_data, imu_time_stamps = ld.load_omega_acc(data_number)
    # Load Vicon data
    if have_vicon:
        vicon_rots, vicon_time_stamps = ld.load_vicon(data_number)
    # Load camera images
    if generate_video:
        cam_imgs, cam_time_stamps = ld.load_camera_image(data_number)

    Rots, time_stamps = UKF.sequential_filtering(omega_data, acc_data, imu_time_stamps)

    # Generate panorama stitching video
    if generate_video:
        # Record valid number of images captured by camera
        k = cam_time_stamps.shape[1]
        k_est = time_stamps.shape[1]

        video_name = "panorama_" + str(data_number) + ".avi"

        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (W, H))

        est_time_index = 0
        for i in range(0, k - 50):
            camera_time = cam_time_stamps[0, i]
            est_time = time_stamps[0, est_time_index]

            while est_time < camera_time:
                est_time_index += 1
                if est_time_index >= k_est:
                    est_time_index -= 1
                    est_time = time_stamps[0, est_time_index]
                    break
                est_time = time_stamps[0, est_time_index]

            img = cam_imgs[:, :, :, i]
            Rot = Rots[:, :, est_time_index]

            canvas = pano.stitch_one_img(img, Rot, canvas, ratio)
            out.write(canvas[:, :, ::-1])       # Flip the channels since OpenCV uses BGR order

        out.release()
        plt.imsave("panorama_" + str(data_number) + ".png", canvas)

    # Gyro only estimation
    Rots_gyro, time_stamps_gyro = UKF.sequential_movements(omega_data, imu_time_stamps)

    # Accelerometer only estimation
    # Use low pass filter on the raw accelerometer data
    acc_data_filtered = UKF.low_pass_filtering(acc_data, alpha=0.7)
    # Calculate pitch and roll angle
    pitch, roll = UKF.pitch_roll_from_acc(acc_data_filtered)

    if have_vicon:
        UKF.plot_with_vicon(Rots_est=Rots,
                            time_stamps_est=time_stamps,
                            Rots_gyro=Rots_gyro,
                            roll=roll,
                            pitch=pitch,
                            time_stamps_imu=time_stamps_gyro,
                            vicon_rots=vicon_rots,
                            vicon_time_stamps=vicon_time_stamps,
                            data_number=data_number,
                            show_figure=False)
    else:
        UKF.plot_without_vicon(Rots_est=Rots,
                               time_stamps_est=time_stamps,
                               Rots_gyro=Rots_gyro,
                               roll=roll,
                               pitch=pitch,
                               time_stamps_imu=time_stamps_gyro,
                               data_number=data_number,
                               show_figure=False)

    return Rots, time_stamps



if __name__ == "__main__":

    main(data_number=8, generate_video=False, have_vicon=True)
