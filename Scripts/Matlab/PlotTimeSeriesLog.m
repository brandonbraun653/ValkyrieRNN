%% Variable Initialization
clear all;
addpath('C:\git\GitHub\ValkyrieRNN\DroneData');

logFilename = 'timeSeriesData.csv';

flightData = struct();

TICK_INDEX = 1;
PITCH_INDEX = 2;
ROLL_INDEX = 3;
YAW_INDEX = 4;
M1_INDEX = 5;
M2_INDEX = 6;
M3_INDEX = 7;
M4_INDEX = 8;

%% Read in data
rawData = csvread(logFilename,1,0);

flightData.time = rawData(:,TICK_INDEX);
flightData.pitch = rawData(:,PITCH_INDEX);
flightData.roll = rawData(:,ROLL_INDEX);
flightData.yaw = rawData(:,YAW_INDEX);
flightData.m1 = rawData(:,M1_INDEX);
flightData.m2 = rawData(:,M2_INDEX);
flightData.m3 = rawData(:,M3_INDEX);
flightData.m4 = rawData(:,M4_INDEX);

%% Plot the data

%ANGLES -------------------------------------
figure(1); clf(1);
subplot(3,1,1); grid on;
plot(flightData.time, flightData.pitch);
title('Measured Pitch Angle');
xlabel('freeRTOS Tick');
ylabel('Angle (deg)');

subplot(3,1,2); grid on;
plot(flightData.time, flightData.roll);
title('Measured Roll Angle');
xlabel('freeRTOS Tick');
ylabel('Angle (deg)');

subplot(3,1,3); grid on;
plot(flightData.time, flightData.yaw);
title('Measured Yaw Angle');
xlabel('freeRTOS Tick');
ylabel('Angle (deg)');

%MOTOR COMMANDS ------------------------------
figure(2); clf(2);
hold on; grid on;
plot(flightData.time, flightData.m1, '-');
plot(flightData.time, flightData.m2, '-');
plot(flightData.time, flightData.m3, '-');
plot(flightData.time, flightData.m4, '-');
title('Motor Command Signals');
xlabel('freeRTOS Tick');
ylabel('PWM Hi (uS)');
legend('M1', 'M2', 'M3', 'M4');

