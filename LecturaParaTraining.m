clc;
close all;
addpath('./');  

%% SELECT THE WORKING DIRECTORY
workingDirectory = uigetdir(pwd, ...
    'Select thew folder where you want to store the extracted .png files');
cd(workingDirectory);

%% CREATE RESULT DIRECTORY
mkdir png
cd('./png');
    
%% READ THE SOURCE FOLDER
sourceDirectory = uigetdir('../../../../../../../../MATLAB/DATA MINING/COVID/', 'Select the .dcm dataset folder');
root = dir(fullfile(sourceDirectory));
for folderIndex = 1 : length(dirList)
    if (length(root(folderIndex).name) > 3)  
        %% READ THE FOLDER THAT ACTUALLY CONTAINS A .dcm FILE
        path = append(root(folderIndex).folder, '\', root(folderIndex).name);
        son = dir(path);
        sonPath = append(son(3).folder, '\', son(3).name);
        grandSon = dir(sonPath);
        container = append(grandSon(3).folder, '\', grandSon(3).name);
        dcmFiles = dir(fullfile(container, '*.dcm'));
        fprintf('--- Directory: %s ---\n', grandSon(3).name);
        fprintf('Number of files in the current directory: %d\n', length(dcmFiles));
        
        %% PROCESS EACH FILE
        mkdir(grandSon(3).name);
%         for fileIndex = 1: 1: length(dcmFiles)
%             
%             masked = dcm2masked(fullfile(dcmFiles(fileIndex).folder, ...
%                 dcmFiles(fileIndex).name));
%             
%             %% SAVE FILE
%             pngFileName = append(extractBetween(dcmFiles(fileIndex).name, ...
%                 1, strlength(dcmFiles(fileIndex).name) -4), '.png');
%             
%             % Write Image
%             imwrite(masked, char(pngFileName));
% 
%             % Move Images To Folder
%             movefile(char(pngFileName), grandSon(3).name);
%             fprintf('Image %d/%d \n', fileIndex, length(dcmFiles))
%         end
    end
end
fprintf('<<< All files have been processed >>>\n');