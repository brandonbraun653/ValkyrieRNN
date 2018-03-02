function smoothData = PreprocessLogData(csv_path)
    addpath(csv_path)
    
    inputLogFilename = 'timeSeriesDataInterpolated.csv';
    
end


%% Variable Initialization
clear all;
csvPath = 'C:\git\GitHub\ValkyrieRNN\DroneData\csv\';
addpath(csvPath);



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
