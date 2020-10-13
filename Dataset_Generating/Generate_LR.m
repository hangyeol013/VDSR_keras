dataset = '91_images';
folder = ['Train/', dataset];
%folder = ['Test/', dataset];
folder_hr = [folder, '/HR'];

%% initialization
filepaths = dir(fullfile(folder_hr,'*.png'));
    
%% for

for i = 1 : length(filepaths)
    
    for scale = 2:4
        
        image_path_lr = [folder, '/LR', '/scale', num2str(scale)];
        %image_path_hr_crop = ['Train/HR_crop/', dataset];
    
        image = imread(fullfile(folder_hr,filepaths(i).name));

        hr_rgb = im2double(image);
        hr_rgb = modcrop(hr_rgb, scale);

        lr_rgb = imresize(hr_rgb, 1/scale, 'bicubic');

        %imName_hr_rgb = [image_path_hr_crop, '/hr_RGB_', num2str(i), '.png'];
        imName_lr_rgb = [image_path_lr, '/lr_RGB_', num2str(scale), '_', num2str(i), '.png'];

        %imwrite(hr_rgb, imName_hr_rgb);
        imwrite(lr_rgb, imName_lr_rgb);

    end
    
end