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
    joints = heatmapTo3DJoints(heatmaps,size(test_3D_joint,1));
    
    %%%  estimate float maximum location  @cf
    sigma = [0.15 0 0;0 0.15 0; 0 0 0.15];
    alldis = zeros(17,17,17,numKps);
    for j = 1:numKps
        joint = joints(:,j);
        x = joint(1)-1:0.125:joint(1)+1;
        y = joint(2)-1:0.125:joint(2)+1;
        z = joint(3)-1:0.125:joint(3)+1;
        mu = [joint(1) joint(2) joint(3)];
        [X,Y,Z] = meshgrid(x,y,z);
        grid = [X(:) Y(:) Z(:)];
        heatmap = mvnpdf(grid,mu,sigma);
        heatmap = reshape(heatmap,[17,17,17]);
        heatmap = double(heatmap);
        
        map_test = heatmaps(:,:,:,j);
        map_to_conv = zeros(17,17,17);
        for k = 1:17
            for m = 1:17
                for n = 1:17
                    if (k==1||k==17||k==9) &&(m==1||m==17||m==9)&&(n==1||n==17||n==9)
                        map_to_conv(k,m,n) = map_test(joint(1)-((9-k)/8),joint(2)-((9-m)/8),joint(3)-((9-n)/8));
                    else
                        map_to_conv(k,m,n) = interp3(map_test,joint(1)-((9-k)*0.125),joint(2)-((9-m)*0.125),joint(3)-((9-n)*0.125));
                    end
                end
            end
        end
              
        
        dis = convn(heatmap, map_to_conv,'same');
        alldis(:,:,:,j) = dis;  
    end
    delta = heatmapTo3DJoints(alldis,17);
    new_joints = joints+(9-delta)*0.125;
    Sall(img_i,:,:) = new_joints;
% 	Sall(img_i,:,:) = joints;
% 
end
% load('Sall.mat');
test_3D_joint = permute(test_3D_joint,[3,2,1]);
test_3D_joint = 76.*test_3D_joint;
Sall = 76.*Sall;
Print results in file H36M.txt
errorH36M(Sall,test_3D_joint);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%compute the int operator error for groundtruth
% S = zeros(Nimg,3,17);
% for i = 1:Nimg
%     heatmaps = zeros(64,64,64,17);
%     for j = 1:17
%         mu = [test_3D_joint(i,1,j) test_3D_joint(i,2,j) test_3D_joint(i,3,j)];
%         sigma = [4 0 0;0 4 0; 0 0 4];
%         [x y z ] = meshgrid(linspace(1,64,64),linspace(1,64,64),linspace(1,64,64));
%         X = [x(:) y(:) z(:)];
%         heatmap = mvnpdf(X,mu,sigma);
%         heatmap = reshape(heatmap,[64,64,64]);

%         heatmaps(:,:,:,j) = heatmap;
%     end
%     joints = heatmapTo3DJoints(heatmaps,17);
%     joints = [joints(2,:); joints(3,:);joints(1,:)];
%     S(i,:,:) = joints;
% end
% load('S.mat');
% test_3D_joint = 76*test_3D_joint;
% S = 76*S;
% errorH36M(S,test_3D_joint);
