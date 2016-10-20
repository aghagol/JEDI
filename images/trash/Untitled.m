clc
clear

image = imread('grass.png');
image = cat(1,image,image(end,:,:),image(end,:,:));

imwrite(image,'grass.png');