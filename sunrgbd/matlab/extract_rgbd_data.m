%% Dump SUNRGBD data to our format
% for each sample, we have RGB image, 2d boxes, point cloud (in camera
% coordinate), calibration and 3d boxes
%
% Author: Charles R. Qi
% Date: 09/27/2017
%
clear; close all; clc;

PROJECT_DIR = '/home/xxx/frustum-convnet/'
SUNRGBD_ROOT = strcat(PROJECT_DIR, 'data/sunrgbd/SUNRGBD/') % ensure it has directories like "kv1  kv2  realsense  xtion"
SUNRGBDTOOL_ROOT = strcat(PROJECT_DIR, 'data/sunrgbd/SUNRGBDtoolbox/')
SAVE_ROOT = strcat(PROJECT_DIR, 'sunrgbd/mysunrgbd/training/')
SET_ROOT = strcat(PROJECT_DIR, 'sunrgbd/image_sets')

% addpath(genpath('.'));
addpath(genpath(SUNRGBDTOOL_ROOT));

load(strcat(SUNRGBDTOOL_ROOT, 'Metadata/SUNRGBDMeta.mat'));

pc_folder = strcat(SAVE_ROOT, 'pc/');
depth_folder = strcat(SAVE_ROOT, 'depth/');
image_folder = strcat(SAVE_ROOT, 'image/');
calib_folder = strcat(SAVE_ROOT, 'calib/');
label_folder = strcat(SAVE_ROOT, 'label/');

mkdir(pc_folder);
mkdir(depth_folder);
mkdir(image_folder);
mkdir(calib_folder);
mkdir(label_folder);

% write imageset file
% official defination is in SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat, same as follow code
% fid=fopen(strcat(SET_ROOT, 'val.txt'),'wt');
% for i=1:5050
%     fprintf(fid,'%06d\n',i);
% end
% fclose(fid);

% fid=fopen(strcat(SET_ROOT, 'train.txt'),'wt');
% for i=5051:10335
%     fprintf(fid,'%06d\n',i);
% end
% fclose(fid);

%% Read
% parpool('local', 10);
% parfor imageId = 1:10335
for imageId = 1:10335
    disp(imageId);
    data = SUNRGBDMeta(imageId);
    data.depthpath(1:25) = ''; % /n/fs/sun3d/data/SUNRGBD = ''
    data.depthpath = strcat(SUNRGBD_ROOT, data.depthpath);
    data.rgbpath(1:25) = '';
    data.rgbpath = strcat(SUNRGBD_ROOT, data.rgbpath);
    % Write point cloud in depth map
    [rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
    rgb(isnan(points3d(:,1)),:) = [];
    points3d(isnan(points3d(:,1)),:) = [];
    points3d_rgb = [points3d, rgb];
    filename = strcat(num2str(imageId,'%06d'), '.txt');
    % numpy is very slow to read large txt file
    % dlmwrite(strcat(depth_folder, filename), points3d_rgb, 'delimiter', ' ');
    matname = strcat(num2str(imageId,'%06d'), '.mat');
    parsave(strcat(pc_folder, matname), points3d_rgb);

    % Write images
    copyfile(data.rgbpath, sprintf('%s/%06d.jpg', image_folder, imageId));
    copyfile(data.depthpath, sprintf('%s/%06d.png', depth_folder, imageId));

    % Write calibration
    dlmwrite(strcat(calib_folder, filename), data.Rtilt(:)', 'delimiter', ' ');
    dlmwrite(strcat(calib_folder, filename), data.K(:)', 'delimiter', ' ', '-append');

    % Write 2D and 3D box label
    % data2d = SUNRGBDMeta2DBB(imageId);
    data2d = data;
    fid = fopen(strcat(label_folder, filename), 'w');
    for j = 1:length(data.groundtruth3DBB)
        %if data2d.groundtruth2DBB(j).has3dbox == 0
        %    continue
        %end
        centroid = data.groundtruth3DBB(j).centroid;
        classname = data.groundtruth3DBB(j).classname;
        orientation = data.groundtruth3DBB(j).orientation;
        coeffs = abs(data.groundtruth3DBB(j).coeffs);
        [new_basis, new_coeffs] = order_basis(data.groundtruth3DBB(j).basis, coeffs, centroid);
        box2d = data2d.groundtruth2DBB(j).gtBb2D;
        %assert(strcmp(data2d.groundtruth2DBB(j).classname, classname));
        if length(box2d) == 0
            continue
        end
        fprintf(fid, '%s %d %d %d %d %f %f %f %f %f %f %f %f %f %f %f %f\n', ...
            classname, box2d(1), box2d(2), box2d(3), box2d(4), ...
            centroid(1), centroid(2), centroid(3), ...
            coeffs(1), coeffs(2), coeffs(3), ...
            new_basis(1,1), new_basis(1,2), new_basis(2,1), new_basis(2,2), ...
            orientation(1), orientation(2));
    end
    fclose(fid);
end
