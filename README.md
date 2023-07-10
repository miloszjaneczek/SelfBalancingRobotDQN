# SelfBalancingRobotDQN
To train the algorithm:
1. Flash the firmware onto the Arduino
2. Make sure that the correct serial port is used
3. Make sure that the Arduino is held in a perfect upright position (the sensors will calibrate) and press its reset button while simultaneously launching the DQN.py file on your computer
4. When the calibration is complete (once you hear the motors running slightly) set the robot on a flat surface - the learning will now begin
To use the algorithm:
1. Add epsilon = 0 line at line 92 in DQN.py
2. Change the value of the pidToggle variable in firmware to for example 60 (to prevent PID from activating)
3. Follow the steps for training the algorithm
