%% Variable Initialization
addpath('C:\git\GitHub\ValkyrieRNN\DroneData\csv');

filename = 'motorLog.csv';

motorData = struct('time', [], 'm1', [], 'm2', [], 'm3', [], 'm4', []);

TICK_INDEX = 1;
M1 = 2;
M2 = 3;
M3 = 4;
M4 = 5;

%% Read in data
inputData = csvread(filename);

motorData.time = inputData(:,TICK_INDEX);
motorData.m1 = inputData(:,M1);
motorData.m2 = inputData(:,M2);
motorData.m3 = inputData(:,M3);
motorData.m4 = inputData(:,M4);


%% Plot the data

figure(1); clf(1);
hold on; grid on;
plot(motorData.m1(:,1), '-');
plot(motorData.m2(:,1), '-');
plot(motorData.m3(:,1), '-');
plot(motorData.m4(:,1), '-');
title('Motor Command Signals');
xlabel('Samples');
ylabel('PWM Hi (uS)');
legend('M1', 'M2', 'M3', 'M4');