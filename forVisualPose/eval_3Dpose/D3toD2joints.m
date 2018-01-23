clear; startup;
opt.visualise = true;
% datapath = '/home/chenf/Documents/pose_estimation/data/multiviewH36m17j/test/c2/';
% predpath =  '/media/chenf/My Passport/';
% annotfile = '/home/chenf/Documents/pose_estimation/data/multiviewH36m17j/test_3D_joint.mat';
% load(annotfile);
% Nimg = size(test_3D_joint,3);
load('Sall.mat');
% Sall = permute(test_3D_joint,[3,2,1]);
load('/home/chenf/Documents/pose_estimation/data/multiviewH36m17j/camera1.mat');
load('/home/chenf/Documents/pose_estimation/data/multiviewH36m17j/camera2.mat');
load('/home/chenf/Documents/pose_estimation/data/multiviewH36m17j/camera3.mat');
load('/home/chenf/Documents/pose_estimation/data/multiviewH36m17j/camera4.mat');

R1 = [camera1(1),camera1(2),camera1(3);camera1(4),camera1(5),camera1(6);camera1(7),camera1(8),camera1(9)];
T1 = [camera1(10);camera1(11);camera1(12)];
k1 = [camera1(17);camera1(18);camera1(19)];
f1 = [camera1(13),camera1(14)];
c1 = [camera1(15),camera1(16)];

R2 = [camera2(1),camera2(2),camera2(3);camera2(4),camera2(5),camera2(6);camera2(7),camera2(8),camera2(9)];
T2 = [camera2(10);camera2(11);camera2(12)];
k2 = [camera2(17);camera2(18);camera2(19)];
f2 = [camera2(13),camera2(14)];
c2 = [camera2(15),camera2(16)];

R3 = [camera3(1),camera3(2),camera3(3);camera3(4),camera3(5),camera3(6);camera3(7),camera3(8),camera3(9)];
T3 = [camera3(10);camera3(11);camera3(12)];
k3 = [camera3(17);camera3(18);camera3(19)];
f3 = [camera3(13),camera3(14)];
c3 = [camera3(15),camera3(16)];

R4 = [camera4(1),camera4(2),camera4(3);camera4(4),camera4(5),camera4(6);camera4(7),camera4(8),camera4(9)];
T4 = [camera4(10);camera4(11);camera4(12)];
k4 = [camera4(17);camera4(18);camera4(19)];
f4 = [camera4(13),camera4(14)];
c4 = [camera4(15),camera4(16)];

Nimg = size(Sall,1);
joints_out = zeros(2,17,67971);
% figure;
for i = 4944:Nimg
    joints = squeeze(Sall(i,:,:));
    joints = joints/76;
    N = size(joints,2);
    X = R4*(joints-T4*ones(1,N));
    XX = X(1:2,:)./([1;1]*X(3,:));
    r2 = XX(1,:).^2+XX(2,:).^2;
    radial = 1 + dot(repmat(k4,[1 N]), [r2; r2.^2; r2.^3], 1);
    tan = camera4(20)*XX(2,:) + camera4(21)*XX(1,:);    
    XXX = XX.*repmat(radial+tan,[2 1]) + [camera4(21) camera4(20)]'*r2;
    Proj = ones(N,1)*f4 .* XXX' + ones(N,1)*c4;
    projJoints = Proj;
    projJoints = permute(projJoints,[2,1]);
    if opt.visualise
        img = imread(['/home/chenf/Documents/pose_estimation/data/multiviewH36m17j/test/c4/im',num2str(i),'.jpg']);
        figure(1),imshow(uint8(img));
        hold on
        
        plotSkeleton_14(projJoints,[],[]);
        hold off
    end
    if opt.visualise; waitforbuttonpress; end
    joints_out(:,:,i) = projJoints;
end
save('joints4.mat','joints_out');
         