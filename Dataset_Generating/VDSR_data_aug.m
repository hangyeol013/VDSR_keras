clear; 
%% To do data augmentation
folder = 'Train/BSD200';
savepath = 'Train/BSD200_aug/';


filepaths = dir(fullfile(folder,'*.jpg'));
     
for i = 1 : length(filepaths)
    filename = filepaths(i).name;
    [add, im_name, type] = fileparts(filepaths(i).name);
    image = imread(fullfile(folder, filename));
    
    for angle = 0: 90 :270
        im_rot = imrotate(image, angle);
        %imwrite(im_rot, [savepath im_name, '_rot' num2str(angle) '.bmp']);
        
        for downsize = 0.6:0.2:1.0
            im_down = imresize(im_rot, downsize, 'bicubic');
            %imwrite(im_down, [savepath im_name, '_rot' num2str(angle) '_s' num2str(downsize) '.bmp']);
            
            for flip = 1:3
                
                if flip == 1
                    image = flipdim(im_down, 1);
                    imwrite(image, [savepath im_name, '_rot' num2str(angle) '_s' num2str(downsize) '_hflip' '.bmp']);
                end
                
                if flip == 2
                    image = flipdim(im_down, 2);
                    imwrite(image, [savepath im_name, '_rot' num2str(angle) '_s' num2str(downsize) '_vflip' '.bmp']);
                end
                
                if flip == 3
                    image = im_down;
                    imwrite(image, [savepath im_name, '_rot' num2str(angle) '_s' num2str(downsize) '_original' '.bmp']);
                end
            end
        end
        
    end
end
