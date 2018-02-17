%% Variable Initialization
addpath('C:\git\GitHub\ValkyrieRNN\DroneData');

filename = 'ahrsLogMinimal.csv';

ahrsData = struct('time', [], 'pitch', [], 'roll', [], 'yaw', []);

TICK_INDEX = 1;
PITCH_INDEX = 2;
ROLL_INDEX = 3;
YAW_INDEX = 4;

%% Read in data
fullRawData = csvread(filename);

ahrsData.time = fullRawData(:,TICK_INDEX);
ahrsData.pitch = fullRawData(:,PITCH_INDEX);
ahrsData.roll = fullRawData(:,ROLL_INDEX);
ahrsData.yaw = fullRawData(:,YAW_INDEX);


%% Plot the data

%ANGLES -------------------------------------
figure(1); clf(1);
subplot(3,1,1); grid on;
plot(ahrsData.pitch(:,1));
title('Measured Pitch Angle');
xlabel('Samples');
ylabel('Angle (deg)');

subplot(3,1,2); grid on;
plot(ahrsData.roll(:,1));
title('Measured Roll Angle');
xlabel('Samples');
ylabel('Angle (deg)');

subplot(3,1,3); grid on;
plot(ahrsData.yaw(:,1));
title('Measured Yaw Angle');
xlabel('Samples');
ylabel('Angle (deg)');

