%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%test whether the interp method is valid, use groundtruth to interp and
%compare with the original groundtruth
%just use one picture for example
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

S = zeros(Nimg,3,numKps);
for i = 1:Nimg
    heatmaps = zeros(64,64,64,17);
    for j = 1:17
        mu = [test_3D_joint(j,1,i) test_3D_joint(j,2,i) test_3D_joint(j,3,i)];
        sigma = [4 0 0;0 4 0; 0 0 4];
        [x y z ] = meshgrid(linspace(1,64,64),linspace(1,64,64),linspace(1,64,64));
        X = [x(:) y(:) z(:)];
        heatmap = mvnpdf(X,mu,sigma);
        heatmap = reshape(heatmap,[64,64,64]);

        heatmaps(:,:,:,j) = heatmap;
    end
    joints = heatmapTo3DJoints(heatmaps,17);
    joints = [joints(2,:); joints(3,:);joints(1,:)];
    
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
%     joints = [joints(2,:); joints(3,:);joints(1,:)];
    S(i,:,:) = new_joints;
    
    S1 = test_3D_joint(:,:,1);
    S1 = S1';
    S1 = S1*76;
    S2 = joints*76;
    S3 = new_joints*76;
    S1 = S1 - repmat(S1(:,1),1,size(S1,2));
    S2 = S2 - repmat(S2(:,1),1,size(S2,2));
    S3 = S3 - repmat(S3(:,1),1,size(S3,2));
        % mean per joint 3D error
    dist1 = mean(sqrt(sum((S1-S2).^2,1)));
    
    dist2 = mean(sqrt(sum((S1-S3).^2,1)));
end