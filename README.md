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
![Panorama stitching](https://lh3.googleusercontent.com/q1vnNtcft0Jbm1kivRN6OkxAzyXRJgLkM9Lzt_CcbWl1B1h_TiFEw7sed2wjJNDTzXlCRy9My1aU4xCmpoLELZn9yZHhV3Ci733y-rAGumxvf-_19JBSxB53Ndf57z1nGkaspx-kBZoDMqGyHSGcmvAakI1jt4SiBmsWmQ4uU00iWFDJT82A4eZUvW-NG6litoHoTFQXJ-9ngXCuEbAWAnCumqX1lCFNgJ22ucSl1ZCzyPGszZe1T3VHq6zM3JSKotLVYi3pal12HlQ9knH3S84FXj3o1kvV8rRU8qoFo6Htae3uKthECWKLnmlvbLPsifbIUQpgbJEgJmPeJuGFDoEv-KgFT_Q2umGhzHzGsXpDCAHA3z1Vo7AmGE-OZtN16NYoXvK-9BkUv78sjicKdUWoIybDUiuB24_LmIGB6k52dkXQo1NhOrsSgegOs-O2jlmYN3BwpRz3JtoM-mD3Rue469ehoWcUH_Sdxb9UTN_gsYU76Vr71-iBGmlk2FaCnPib-jHjQCi8KSP9PgFHy4gdTcD4p2_Ak_aHxV88xtYxakZ5iFEqcdbF1EHGgVClCZCgqo7XAwcfm2sDwW340dsbqcdo39W_9Rt9Wg=w640-h480-no)
