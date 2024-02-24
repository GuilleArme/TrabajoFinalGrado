matfiles = dir('*.mat');
resizeCoef = 2.5;
mkdir resized;
mkdir resized\test_data
mkdir resized\test_data\images;
mkdir resized\test_data\ground-truth;
mkdir resized\train_data
mkdir resized\train_data\images;
mkdir resized\train_data\ground-truth;
i = 1;
itrain = 1;
itest = 1;
DataSize = size(matfiles,1);
r= randperm(DataSize, ceil(DataSize*0.2));
for file = matfiles'
    load(file.name, "turbotCentroidS"); 
    image_info{1,1}.location = turbotCentroidS(:,2:3)/ resizeCoef;
    image_info{1,1}.number = size(turbotCentroidS,1);
    SNameBase = erase(file.name, "_c.mat");
    sNameImage = strcat(SNameBase,".jpg");
    I = imread(sNameImage);
    B = imresize(I,(1/resizeCoef));
    if find(i==r)
        newImageName = strcat('IMG_', string(itest), ".jpg");
        newMatName = strcat("GT_IMG_",  string(itest), ".mat");
        imwrite(B,strcat("./resized/test_data/images/", newImageName), "jpg");
        save(strcat("./resized/test_data/ground-truth/", newMatName),'image_info');
        itest = itest +1;
    else
        newImageName = strcat('IMG_', string(itrain), ".jpg");
        newMatName = strcat("GT_IMG_",  string(itrain), ".mat");
        imwrite(B,strcat("./resized/train_data/images/", newImageName), "jpg");
        save(strcat("./resized/train_data/ground-truth/", newMatName),'image_info');
        itrain = itrain +1;
    end
    i = i + 1;
end

