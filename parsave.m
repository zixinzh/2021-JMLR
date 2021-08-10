function [ output_args ] = parsave( filename, x1, x2, x3, x4, x5, x6, x7 )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%     save(filename, 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7');
    
    if nargin == 2
        save(filename, 'x1');
    end
    
    if nargin == 3
        save(filename, 'x1', 'x2');
    end
    
    if nargin == 4
        save(filename, 'x1', 'x2', 'x3');
    end
    
    if nargin == 5
        save(filename, 'x1', 'x2', 'x3', 'x4');
    end

    if nargin == 6
        save(filename, 'x1', 'x2', 'x3', 'x4', 'x5');
    end
    
    if nargin == 7
        save(filename, 'x1', 'x2', 'x3', 'x4', 'x5', 'x6');
    end
    
    if nargin == 8
        save(filename, 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7');
    end

end

