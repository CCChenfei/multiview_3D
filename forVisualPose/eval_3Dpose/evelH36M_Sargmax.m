%% Evaluation for Human3.6M full test set
% We assume that the network has already been applied on the Human3.6M sample images.
% This code reads the network predictions and estimates metric 3D pose 
% from the volumetric output. Full results for the Human3.6M dataset 
% (subjects S9 and S11) are printed in file H36M.txt

clear; startup;

% define paths for data and predictions
datapath = '/home/chenf/Documents/pose_estimation/data/multiviewH36m17j/test/c2/';
predpath =  '/home/chenf/PycharmProjects/multiview_3D/';
annotfile = '/home/chenf/Documents/pose_estimation/data/multiviewH36m17j/test_3D_joint.mat';
load(annotfile);
Nimg = size(test_3D_joint,3);
% define the reconstruction from the volumetric representation
% if recType = 1, we use the groundtruth depth of the root joint
% if recType = 2, we estimate the root depth based on the subject's skeleton size
% if recType = 3, we estimate the root depth based on the training subjects' mean skeleton size
recType = 3;

% volume parameters
outputRes = 64;     % x,y resolution
depthRes = 64;      % z resolution
numKps = 17;        % number of joints

% Recover 3D predictions
Sall = zeros(Nimg,3,numKps);
for img_i = 1:Nimg
    heatmaps = hdf5read([predpath 'out_heatmap.h5'],['/out3D',num2str(img_i)]);
    heatmaps = permute(heatmaps,[4,3,2,1]);
    % pixel location
    joints = heatmapToJoints_Sargmax(heatmaps,size(test_3D_joint,1));
%     joints = permute(joints,[3,2,1]);
	Sall(img_i,:,:) = joints;
end
% % load('Sall.mat');
test_3D_joint = permute(test_3D_joint,[3,2,1]);
% test_3D_joint = test_3D_joint+1;
test_3D_joint = 76.*test_3D_joint;
Sall = 76.*Sall;
% load('interp_test_joint_afterenlarge.mat');
% print 'results in file H36M.txt';
errorH36M(Sall,test_3D_joint);
