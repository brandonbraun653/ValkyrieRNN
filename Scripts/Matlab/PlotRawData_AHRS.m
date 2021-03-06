%% Variable Initialization
addpath('C:\git\GitHub\ValkyrieRNN\DroneData');

flightLogFilename = 'ahrsLogFull.csv';

ahrsData = struct(...
    'pitch', [], 'roll', [], 'yaw', [],...
    'ax', [], 'ay', [], 'az', [],...
    'gx', [], 'gy', [], 'gz', [],...
    'mx', [], 'my', [], 'mz', []);
    
PITCH_INDEX = 1;
ROLL_INDEX = 2;
YAW_INDEX = 3;
AX_INDEX = 4;
AY_INDEX = 5;
AZ_INDEX = 6;
GX_INDEX = 7;
GY_INDEX = 8;
GZ_INDEX = 9;
MX_INDEX = 10;
MY_INDEX = 11;
MZ_INDEX = 12;


%% Read in data
fullRawData = csvread(flightLogFilename);

ahrsData.pitch = fullRawData(:,PITCH_INDEX);
ahrsData.roll = fullRawData(:,ROLL_INDEX);
ahrsData.yaw = fullRawData(:,YAW_INDEX);
ahrsData.ax = fullRawData(:,AX_INDEX);
ahrsData.ay = fullRawData(:,AY_INDEX);
ahrsData.az = fullRawData(:,AZ_INDEX);
ahrsData.gx = fullRawData(:,GX_INDEX);
ahrsData.gy = fullRawData(:,GY_INDEX);
ahrsData.gz = fullRawData(:,GZ_INDEX);
ahrsData.mx = fullRawData(:,MX_INDEX);
ahrsData.my = fullRawData(:,MY_INDEX);
ahrsData.mz = fullRawData(:,MZ_INDEX);

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

%ACCELEROMETER ------------------------------
figure(2); clf(2);
subplot(3,1,1); grid on;
plot(ahrsData.ax(:,1));
title('Ax');
xlabel('Samples');
ylabel('Acceleration (m/s^2)');

subplot(3,1,2); grid on;
plot(ahrsData.ay(:,1));
title('Ay');
xlabel('Samples');
ylabel('Acceleration (m/s^2)');

subplot(3,1,3); grid on;
plot(ahrsData.az(:,1));
title('Az');
xlabel('Samples');
ylabel('Acceleration (m/s^2)');


%GYROSCOPE ---------------------------------
figure(3); clf(3);
subplot(3,1,1); grid on;
plot(ahrsData.gx(:,1));
title('Gx');
xlabel('Samples');
ylabel('Rotation Rate (dps)');

subplot(3,1,2); grid on;
plot(ahrsData.gy(:,1));
title('Gy');
xlabel('Samples');
ylabel('Rotation Rate (dps)');

subplot(3,1,3); grid on;
plot(ahrsData.gz(:,1));
title('Gz');
xlabel('Samples');
ylabel('Rotation Rate (dps)');

%MAGNETOMETER ------------------------------
figure(4); clf(4);
subplot(3,1,1); grid on;
plot(ahrsData.mx(:,1));
title('Mx');
xlabel('Samples');
ylabel('H (gauss)');

subplot(3,1,2); grid on;
plot(ahrsData.my(:,1));
title('My');
xlabel('Samples');
ylabel('H (gauss)');

subplot(3,1,3); grid on;
plot(ahrsData.mz(:,1));
title('Mz');
xlabel('Samples');
ylabel('H (gauss)');

