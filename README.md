## Quaternion-based UKF for Orientation Estimation

- Author: Dian Chen
- Project 2 for ESE650 2018 spring, University of Pennsylvania
- Platform: macOS Sierra 10.12

### General Description
-----
- The generated pictures and videos are uploaded in this [google drive](https://drive.google.com/drive/folders/1ZaK1VYg9BuCkj78rUt05ci320jCe0mOj?usp=sharing)

	List of scripts:
	1. main.py          - main script
	2. load_data.py     - all functions for data loading, preprocessing
	3. UKF.py           - all functions for UKF steps and plots of roll, pitch, yaw curves
	4. panorama.py      - used for generating panorama videos
	5. quaternion.py    - library for quaternion operations

- IMPORTANT: To run the test, please make sure that you’ve put cam, imu, vicon data in correponding subfolders just like the released project package. Here I've included one dataset for you to play with.

- To run the test, simply run the “main.py” script.  There are 4 flags you can play with:
	1. data_number      - the number of dataset to use
	2. generate_video   - whether or not to generate a panorama video (only applicable to datasets with camera data)
	3. show\_gyro_only   - whether or not to compare results with using only gyro
	4. show\_acc_only    - whether or not to compare results with using only accelerometer

### Sample Output
-----
Here's one panorama stitching coming from the UKF estimation:

<p align="center"> 
<img src="https://github.com/DianCh/Quaternion_based_UKF/blob/master/results/panorama_8.png" width="700">
</p>

<p align="center"> 
<img src="https://github.com/DianCh/Quaternion_based_UKF/blob/master/results/Euler_results_8.png" width="400">
</p>)

A detailed report for this project can also be found [here](https://github.com/DianCh/Quaternion_based_UKF/blob/master/results/report.pdf).