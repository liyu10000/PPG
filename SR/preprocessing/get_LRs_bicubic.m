clc; clear

scaling_factors = [2,3,4];
dir_path = '../../SR_dataset/'

read_path = [dir_path, '/HR/'];
files = dir(read_path);
files = files(3:end);
images = cell(length(files),1);
divider = 1;
for is = 1:length(scaling_factors)
    divider = lcm(divider, scaling_factors(is));
end
disp('cutting borders')
for i = 1:length(images)
    hr_image = imread([read_path,files(i).name]);
    [h,w,c] = size(hr_image);
    rh = mod(h,divider);
    rw = mod(w,divider);
    if rh ~= 0
        sh = floor(rh/2) + 1;
        eh = h - (rh - sh) - 1;
        hr_image = hr_image(sh:eh, :, :);
    end
    if rw ~= 0
        sw = floor(rw/2)+1;
        ew = w - (rw - sw) - 1;
         hr_image = hr_image(:, sw:ew, :);
    end
    if (rh ~= 0)||(rw ~= 0)
        disp('writing')
        imwrite(hr_image, [read_path,files(i).name]);
    end
    disp(['finished image ' num2str(i) '/' num2str(length(images))])
end

for is = 1:length(scaling_factors)
    scaling_factor = scaling_factors(is);
    write_path = [dir_path, '/LR/X', num2str(scaling_factor), '/'];
    mkdir(write_path);
    for i = 1:length(images)
        hr_image = imread([read_path,files(i).name]);
        number = extractBefore(files(i).name, 'HR');
        lr_image = imresize(hr_image, 1/scaling_factor);
        imwrite(lr_image, [write_path, number, 'LR.png']);
        disp(['completed image ' num2str(i) '/' num2str(length(images)) ' for scale ' num2str(scaling_factor)])
    end
end
