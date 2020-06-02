%% loadImsFromFolder(root) Method
%  This method loads all the images within a folder and turns them into
%  a grayscale image. Expected size of image is 512x512 px
%  returns: a three-dimesional double array with the first dimension being 
%           the image number and the other two the image data
function ims = loadImsFromFolder(root)
    % [dir('*.jpg');dir('*.png')], .jpg, .jpeg, .jpe .jif, .jfif, .jfi
    imList = [dir(fullfile(root, '*.png')); ... 
        dir(fullfile(root, '*.jpg')); dir(fullfile(root, '*.jpeg')); ...
        dir(fullfile(root, '*.jpe')); dir(fullfile(root, '*.jif')); ...
        dir(fullfile(root, '*.jfif')); dir(fullfile(root, '*.jfi'))];
    images = zeros(length(imList), 512, 512);
    for im = 1: 1: length(imList)
        image = imread(fullfile(imList(im).folder, imList(im).name));
        grayscale = mat2gray(image);
        images(im, :, :) = grayscale;
    end
    ims = images;
end