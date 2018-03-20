function SmoothTimeSeriesData(log_path, output_path, gyro_max, gyro_min, motor_max, motor_min)
%% Variable Initialization
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
motorMin = motor_min;                    %ESC Idle Command
motorMax = motor_max;                    %ESC Max Throttle Command
motorRange = motorMax-motorMin;     

% Gyro Settings
gyroMin = gyro_min; %dps
gyroMax = gyro_max;

%% Process Data
rawData = csvread(log_path,1,0);

% Generate smoothed data for NN model inference testing
inference_smoothedData = [rawData(:,1), smoothdata(rawData(:,2:end))];


% Normalize and scale the motor command data
% motor = rawData(:,M1_INDEX:M4_INDEX) - motorMin;
% motor(motor<0) = 0;
% motor = motor./motorRange;

% Normalize gyro over [-1,1] range and then scale
% gyro = rawData(:,GX_INDEX:GZ_INDEX) + gyroMax;
% gyro = (gyro./gyroMax)-1;

gyro = rawData(:,GX_INDEX:GZ_INDEX);
dc_offset = mean(gyro);
gyro = gyro - dc_offset;


% Recreate the data
norm_filteredData = [rawData(:,TICK_INDEX:AZ_INDEX),...
                     gyro,...
                     rawData(:,MX_INDEX:MZ_INDEX),...
                     rawData(:,M1_INDEX:M4_INDEX),...
                     rawData(:,ASP_INDEX:end)];
   
norm_smoothedData = [norm_filteredData(:,1), smoothdata(norm_filteredData(:,2:end))];

% Write the data to file
scaled_noisyPath = join([output_path,'timeSeriesDataNoisy.csv']);
fid = fopen(scaled_noisyPath, 'w');
fprintf(fid, '%s\n', csvHeader);
fclose(fid);
dlmwrite(scaled_noisyPath, norm_filteredData, '-append');

scaled_smoothPath = join([output_path,'timeSeriesDataSmoothed.csv']);
fid = fopen(scaled_smoothPath, 'w');
fprintf(fid, '%s\n', csvHeader);
fclose(fid);
dlmwrite(scaled_smoothPath, norm_smoothedData, '-append');

inference_smoothPath = join([output_path,'timeSeriesInferenceDataSmoothed.csv']);
fid = fopen(inference_smoothPath, 'w');
fprintf(fid, '%s\n', csvHeader);
fclose(fid);
dlmwrite(inference_smoothPath, inference_smoothedData, '-append');

%% Save some plots to file for user verification
global_res = '-r1200';
font_size = 8;
aspect_ratio = [48 18 1];

%---------------------------------------------
% Plot the AHRS Data
%---------------------------------------------
figure(1); clf(1);

subplot(2,1,1); hold on; grid on;
pbaspect(aspect_ratio);
plot(norm_filteredData(:,PITCH_INDEX));
plot(norm_filteredData(:,ROLL_INDEX));
title('Noisy AHRS Data','FontSize',font_size);
xlabel('Samples','FontSize',font_size);
ylabel('Angle (deg)','FontSize',font_size);
legend('Pitch', 'Roll');

subplot(2,1,2); hold on; grid on;
pbaspect(aspect_ratio);
plot(norm_smoothedData(:,PITCH_INDEX));
plot(norm_smoothedData(:,ROLL_INDEX));
title('Smoothed AHRS Data','FontSize',font_size);
xlabel('Samples','FontSize',font_size);
ylabel('Angle (deg)','FontSize',font_size);
legend('Pitch', 'Roll');

print(join([output_path,'ahrs_plots']), '-djpeg', global_res);

%---------------------------------------------
% Plot the Motor Command Data
%---------------------------------------------
figure(2); clf(2); hold on;
subplot(2,1,1); hold on; grid on;
pbaspect(aspect_ratio);
plot(norm_filteredData(:,M1_INDEX));
plot(norm_filteredData(:,M2_INDEX));
plot(norm_filteredData(:,M3_INDEX));
plot(norm_filteredData(:,M4_INDEX)); 
title('Noisy Motor Command Data','FontSize',font_size);
xlabel('Samples','FontSize',font_size);
ylabel('Scaled Command','FontSize',font_size);
legend('Motor1', 'Motor2', 'Motor3', 'Motor4');

subplot(2,1,2); hold on; grid on;
pbaspect(aspect_ratio);
plot(norm_smoothedData(:,M1_INDEX));
plot(norm_smoothedData(:,M2_INDEX));
plot(norm_smoothedData(:,M3_INDEX));
plot(norm_smoothedData(:,M4_INDEX)); 
title('Smoothed Motor Command Data','FontSize',font_size);
xlabel('Samples','FontSize',font_size);
ylabel('Scaled Command','FontSize',font_size);
legend('Motor1', 'Motor2', 'Motor3', 'Motor4');

print(join([output_path,'motor_plots']), '-djpeg', global_res);

%---------------------------------------------
% Plot the Gyro Data
%---------------------------------------------
figure(3); clf(3); hold on;
subplot(2,1,1); hold on; grid on;
pbaspect(aspect_ratio);
plot(norm_filteredData(:,GX_INDEX));
plot(norm_filteredData(:,GY_INDEX));
plot(norm_filteredData(:,GZ_INDEX));
title('Noisy Gyro Data','FontSize',font_size);
xlabel('Samples','FontSize',font_size);
ylabel('Scaled Measurement','FontSize',font_size);
legend('GX', 'GY', 'GZ');

subplot(2,1,2); hold on; grid on;
pbaspect(aspect_ratio);
plot(norm_smoothedData(:,GX_INDEX));
plot(norm_smoothedData(:,GY_INDEX));
plot(norm_smoothedData(:,GZ_INDEX));
title('Smoothed Gyro Data','FontSize',font_size);
xlabel('Samples','FontSize',font_size);
ylabel('Scaled Measurement','FontSize',font_size);
legend('GX', 'GY', 'GZ');

print(join([output_path,'gyro_plots']), '-djpeg', global_res);

end