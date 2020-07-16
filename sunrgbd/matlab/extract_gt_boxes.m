%% Extract GT boxes

clear all

PROJECT_DIR = '/home/xxx/frustum-convnet/'
SUNRGBD_ROOT = strcat(PROJECT_DIR, 'data/sunrgbd/SUNRGBD/')
SUNRGBDTOOL_ROOT = strcat(PROJECT_DIR, 'data/sunrgbd/SUNRGBDtoolbox/')
SAVE_ROOT='gt_boxes/'

mkdir(SAVE_ROOT);
addpath(genpath(SUNRGBDTOOL_ROOT));

for className = {'bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub'}
    clear('bbs');
    clear('imgids');
    split = load(fullfile(SUNRGBDTOOL_ROOT,'/traintestSUNRGBD/allsplit.mat'));
    testset_path = split.alltest;
    % testset_path{1}
    % for i = 1:length(testset_path)
    %     testset_path{i}(1:25) = ''; % /n/fs/sun3d/data/SUNRGBD = ''
    %     testset_path{i} = strcat(SUNRGBD_ROOT, testset_path{i});
    % end
    [groundTruthBbs,all_sequenceName] = benchmark_groundtruth(className, fullfile(SUNRGBDTOOL_ROOT,'Metadata/'),testset_path);

    nBb = length(groundTruthBbs);
    for i = 1:nBb
        corners = get_corners_of_bb3d(groundTruthBbs(i));
        bbs(i,:) = [reshape([corners(1:4,1) corners(1:4,2)]',1,[]) min(corners([1 end],3)) max(corners([1 end],3))];
        imgids(i) = groundTruthBbs(i).imageNum;
    end
    fprintf('%s:%d\n', className{1}, nBb);
    if nBb > 0
        dlmwrite(strcat(SAVE_ROOT, className{1}, '_gt_boxes.dat'), bbs, 'delimiter', ' ');
        dlmwrite(strcat(SAVE_ROOT, className{1}, '_gt_imgids.txt'), imgids, 'delimiter', ' ');
    end
end