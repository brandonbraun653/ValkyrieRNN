%% Variable Initialization
clear all;
csvPath = 'C:\git\GitHub\ValkyrieRNN\DroneData\csv\';
addpath(csvPath);

inputLogFilename = 'timeSeriesDataInterpolated.csv';

flightData = struct();
header = {'rtosTick', 'pitch', 'roll', 'yaw',... 
          'ax','ay','az','gx','gy','gz','mx','my','mz',...
          'm1CMD', 'm2CMD', 'm3CMD', 'm4CMD',...
          'asp','asr','asy','rsp','rsr','rsy'};
csvHeader = strjoin(header,',');

TICK_INDEX = 1;

%AHRS
PITCH_INDEX = 2;
ROLL_INDEX = 3;
YAW_INDEX = 4;
AX_INDEX = 5;                       %m/s^2
AY_INDEX = 6;
AZ_INDEX = 7;
GX_INDEX = 8;                       %dps
GY_INDEX = 9;
GZ_INDEX = 10;
MX_INDEX = 11;                      %gauss
MY_INDEX = 12;
MZ_INDEX = 13;

% Motor Commands
M1_INDEX = 14;
M2_INDEX = 15;
M3_INDEX = 16;
M4_INDEX = 17;

%PID Controller
ASP_INDEX= 18;                      %Angle Setpoint Pitch
ASR_INDEX= 19;                      %Angle Setpoint Roll
ASY_INDEX= 20;                      %Angle Setpoint Yaw
RSP_INDEX= 21;                      %Rate Setpoint Pitch
RSR_INDEX= 22;                      %Rate Setpoint Roll
RSY_INDEX= 23;                      %Rate Setpoint Yaw

% ESC Settings
motorMin = 1060;                    %ESC Idle Command
motorMax = 1860;                    %ESC Max Throttle Command
motorRange = motorMax-motorMin;     
motorScaleFactor = 10;              %Range to scale to after normalization

% Gyro Settings
gyroMin = -2000; %dps
gyroMax = 2000;
gyroRange = gyroMax-gyroMin;
gyroScaleFactor = 10;

%% Process Data
rawData = csvread(inputLogFilename,1,0);

% Generate smoothed data for NN model inference testing
inference_smoothedData = [rawData(:,1), smoothdata(rawData(:,2:end))];


% Normalize and scale the motor command data
motor = rawData(:,M1_INDEX:M4_INDEX) - motorMin;
motor(motor<0) = 0;
motor = motorScaleFactor*(motor./motorRange);

% Normalize gyro over [-1,1] range and then scale
gyro = rawData(:,GX_INDEX:GZ_INDEX) + gyroMax;
gyro = gyroScaleFactor*((gyro./gyroMax)-1);

% Recreate the data
scaled_filteredData = [rawData(:,TICK_INDEX:AZ_INDEX), gyro,...
                       rawData(:,MX_INDEX:MZ_INDEX), motor,...
                       rawData(:,ASP_INDEX:end)];
   
scaled_smoothedData = [scaled_filteredData(:,1), smoothdata(scaled_filteredData(:,2:end))];

% Write the data to file
scaled_noisyPath = join([csvPath,'timeSeriesDataNoisy.csv']);
fid = fopen(scaled_noisyPath, 'w');
fprintf(fid, '%s\n', csvHeader);
fclose(fid);
dlmwrite(scaled_noisyPath, scaled_filteredData, '-append');

scaled_smoothPath = join([csvPath,'timeSeriesDataSmoothed.csv']);
fid = fopen(scaled_smoothPath, 'w');
fprintf(fid, '%s\n', csvHeader);
fclose(fid);
dlmwrite(scaled_smoothPath, scaled_smoothedData, '-append');

inference_smoothPath = join([csvPath,'timeSeriesInferenceDataSmoothed.csv']);
fid = fopen(inference_smoothPath, 'w');
fprintf(fid, '%s\n', csvHeader);
fclose(fid);
dlmwrite(inference_smoothPath, inference_smoothedData, '-append');


%% Plot if Desired
if 0
    %---------------------------------------------
    % Plot the AHRS Data
    %---------------------------------------------
    figure(1); clf(1);
    subplot(2,1,1); hold on; grid on;
    plot(scaled_filteredData(:,PITCH_INDEX));
    plot(scaled_filteredData(:,ROLL_INDEX));
    title('Noisy AHRS Data');
    xlabel('Samples');
    ylabel('Angle (deg)');
    legend('Pitch', 'Roll');
    
    subplot(2,1,2); hold on; grid on;
    plot(scaled_smoothedData(:,PITCH_INDEX));
    plot(scaled_smoothedData(:,ROLL_INDEX));
    title('Smoothed AHRS Data');
    xlabel('Samples');
    ylabel('Angle (deg)');
    legend('Pitch', 'Roll');
    
    %---------------------------------------------
    % Plot the Motor Command Data
    %---------------------------------------------
    figure(2); clf(2); hold on;
    subplot(2,1,1); hold on; grid on;
    plot(scaled_filteredData(:,M1_INDEX));
    plot(scaled_filteredData(:,M2_INDEX));
    plot(scaled_filteredData(:,M3_INDEX));
    plot(scaled_filteredData(:,M4_INDEX)); 
    title('Noisy Motor Command Data');
    xlabel('Samples');
    ylabel('Scaled Command');
    legend('Motor1', 'Motor2', 'Motor3', 'Motor4');
    
    subplot(2,1,2); hold on; grid on;
    plot(scaled_smoothedData(:,M1_INDEX));
    plot(scaled_smoothedData(:,M2_INDEX));
    plot(scaled_smoothedData(:,M3_INDEX));
    plot(scaled_smoothedData(:,M4_INDEX)); 
    title('Smoothed Motor Command Data');
    xlabel('Samples');
    ylabel('Scaled Command');
    legend('Motor1', 'Motor2', 'Motor3', 'Motor4');
    
    %---------------------------------------------
    % Plot the Gyro Data
    %---------------------------------------------
    figure(3); clf(3); hold on;
    subplot(2,1,1); hold on; grid on;
    plot(scaled_filteredData(:,GX_INDEX));
    plot(scaled_filteredData(:,GY_INDEX));
    plot(scaled_filteredData(:,GZ_INDEX));
    title('Noisy Gyro Data');
    xlabel('Samples');
    ylabel('Scaled Measurement');
    legend('GX', 'GY', 'GZ');
    
    subplot(2,1,2); hold on; grid on;
    plot(scaled_smoothedData(:,GX_INDEX));
    plot(scaled_smoothedData(:,GY_INDEX));
    plot(scaled_smoothedData(:,GZ_INDEX));
    title('Smoothed Gyro Data');
    xlabel('Samples');
    ylabel('Scaled Measurement');
    legend('GX', 'GY', 'GZ');
    
end
