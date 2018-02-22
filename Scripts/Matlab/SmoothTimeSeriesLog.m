%% Variable Initialization
clear all;
addpath('C:\git\GitHub\ValkyrieRNN\DroneData');

logFilename = 'timeSeriesDataRaw.csv';

flightData = struct();
header = {'rtosTick', 'pitch', 'roll', 'yaw', 'm1CMD', 'm2CMD', 'm3CMD', 'm4CMD'};
csvHeader = strjoin(header,',');

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

%% Process Data
fdata = smoothdata(rawData(:,2:end));

%            rtosTick  pitch/roll/yaw          motor commands
nfdata = [rawData(:,1), fdata(:,1:3), 100*normc(fdata(:,4:end)-1060)];

% Write the filtered data to file
filename = 'C:/git/GitHub/ValkyrieRNN/DroneData/timeSeriesDataSmoothed.csv';
fid = fopen(filename, 'w');
fprintf(fid, '%s\n', csvHeader);
fclose(fid);

dlmwrite(filename, nfdata, '-append');

%% Plot if Desired
if 0
    figure(1); clf(1); hold on;
    plot(nfdata(:,PITCH_INDEX));
    plot(nfdata(:,ROLL_INDEX));

    figure(2); clf(2); hold on;
    plot(nfdata(:,M1_INDEX));
    plot(nfdata(:,M2_INDEX));
    plot(nfdata(:,M3_INDEX));
    plot(nfdata(:,M4_INDEX)); 
end
