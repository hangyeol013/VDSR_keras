clear;close all;
%% settings
%% ÇÒ¶§¸¶´Ù folder À§Ä¡ test¶û train ¹Ù²ãÁà¾ß ÇÑ´Ù.
%folder = 'Train/291_images/HR_aug';
folder = 'Train/\BSD200/HR';

savepath_test = 'train_291aug.h5';
savepath_train = 'cross_val_BSD200_new.h5';

size_input = 41;
size_label = 41;
stride = 41;

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);

count = 0;

%% generate data
filepaths = dir(fullfile(folder,'*.jpg'));

for scale = 2:4    
    for i = 1 : length(filepaths)

        image = imread(fullfile(folder,filepaths(i).name));
        image = rgb2ycbcr(image);
        %image = image(:, :, 1);
        image = im2double(image(:, :, 1));

        im_label = modcrop(image, scale);
        [hei,wid] = size(im_label);
        im_lr = imresize(im_label,1/scale,'bicubic');
        im_input = imresize(im_lr, scale, 'bicubic');

        for x = 1 : stride : hei-size_input+1
            for y = 1 :stride : wid-size_input+1

                subim_label = im_label(x : x+size_label-1, y : y+size_label-1);
                subim_input = im_input(x : x+size_input-1, y : y+size_input-1);

                count=count+1;
                
                data(:, :, 1, count) = subim_input;
                label(:, :, 1, count) = subim_label;
            end
        end
    end
end

data_dims = size(data);
label_dims = size(label);

%% writing to HDF5

h5create(savepath_train,'/data', data_dims);
h5create(savepath_train, '/label', label_dims);

h5write(savepath_train, '/data', data);
h5write(savepath_train, '/label', label);

h5disp(savepath_train);
