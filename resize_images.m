
clc;
close all;
clear all;


for i = 71:100
display(i);
str = sprintf('/Users/deepakmuralidharan/sample_test/sample%d',i);
cd(str);
doc = dir('*.jpg');
l = length(doc);
for jj = 1:length(doc)
    im = imread(doc(jj).name);
    im = imresize(im,[224 224]);
    imwrite(im, doc(jj).name);
end
end