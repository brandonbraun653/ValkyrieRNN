function [stepData] = CalculateStepPerformance(y, t, yFinal)
    switch nargin
        case 2
            stepData = stepinfo(y,t);
        case 3
            stepData = stepinfo(y,t,yFinal);
        otherwise
            stepData = 0;
    end
end