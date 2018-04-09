clear;

joints_3D = load('/home/chenf/Documents/pose_estimation/data/human3.6M_ori/joints_3D.mat');
joint = joints_3D.joints_3D{1,1}(:,:,1);
joints_2D = load('/home/chenf/Documents/pose_estimation/data/human3.6M_ori/S1/joints.mat');
joints1 = joints_2D.joints{1}(:,:,1);
joints2 = joints_2D.joints{2}(:,:,1);
joints3 = joints_2D.joints{3}(:,:,1);
joints4 = joints_2D.joints{4}(:,:,1);
% joints1 = joints1(1,:);
% joints2 = joints2(1,:);
% joints3 = joints3(1,:);
% joints4 = joints4(1,:);

I1 = imread('/home/chenf/Documents/pose_estimation/data/human3.6M_ori/S1/c1_2/im1.jpg');
I2 = imread('/home/chenf/Documents/pose_estimation/data/human3.6M_ori/S1/c2_2/im1.jpg');
I3 = imread('/home/chenf/Documents/pose_estimation/data/human3.6M_ori/S1/c3_2/im1.jpg');
I4 = imread('/home/chenf/Documents/pose_estimation/data/human3.6M_ori/S1/c4_2/im1.jpg');

b1 = [-900,-600,15];
b2 = [700,-600,15];
b3 = [-900,-600,1615];
b4 = [700,-600,1615];
b5 = [700,1000,15];
b6 = [-900,1000,15];
b7 = [-900,1000,1615];
b8 = [700,1000,1615];
b = [b1;b2;b3;b4;b5;b6;b7;b8];
a1 = [0,0,0];
a2 = [1600,0,0];
a3 = [0,1600,0];
a4 = [0,0,1600];
a5 = [1600,1600,0];
a6 = [1600,0,1600];
a7 = [0,1600,1600];
a8 = [1600,1600,1600];
a = [a1;a2;a3;a4;a5;a6;a7;a8];
cam1 = load('/home/chenf/Documents/pose_estimation/data/human3.6M_ori/H36MDemo/cam1.mat');
cam2 = load('/home/chenf/Documents/pose_estimation/data/human3.6M_ori/H36MDemo/cam2.mat');
cam3 = load('/home/chenf/Documents/pose_estimation/data/human3.6M_ori/H36MDemo/cam3.mat');
cam4 = load('/home/chenf/Documents/pose_estimation/data/human3.6M_ori/H36MDemo/cam4.mat');
cam1 = cam1.cam1;
cam2 = cam2.cam2;
cam3 = cam3.cam3;
cam4 = cam4.cam4;

root_joint = joint;
root_trans_joint = root_joint-[-900,-600,15];
root_2D_joint1 = ProjectPointRadial(root_trans_joint,cam1(1:9),cam1(10:12),cam1(13:14),cam1(15:16),cam1(17:19),cam1(20:21));
root_2D_joint2 = ProjectPointRadial(root_trans_joint,cam2(1:9),cam2(10:12),cam2(13:14),cam2(15:16),cam2(17:19),cam2(20:21));
root_2D_joint3 = ProjectPointRadial(root_trans_joint,cam3(1:9),cam3(10:12),cam3(13:14),cam3(15:16),cam3(17:19),cam3(20:21));
root_2D_joint4 = ProjectPointRadial(root_trans_joint,cam4(1:9),cam4(10:12),cam4(13:14),cam4(15:16),cam4(17:19),cam4(20:21));

proj1 = ProjectPointRadial(a,cam1(1:9),cam1(10:12),cam1(13:14),cam1(15:16),cam1(17:19),cam1(20:21));
proj2 = ProjectPointRadial(a,cam2(1:9),cam2(10:12),cam2(13:14),cam2(15:16),cam2(17:19),cam2(20:21));
proj3 = ProjectPointRadial(a,cam3(1:9),cam3(10:12),cam3(13:14),cam3(15:16),cam3(17:19),cam3(20:21));
proj4 = ProjectPointRadial(a,cam4(1:9),cam4(10:12),cam4(13:14),cam4(15:16),cam4(17:19),cam4(20:21));

figure;imshow(I1);hold on;plot(root_2D_joint1(:,1),root_2D_joint1(:,2),'go');plot(proj1(:,1),proj1(:,2),'bo');hold off;
figure;imshow(I2);hold on;plot(root_2D_joint2(:,1),root_2D_joint2(:,2),'go');plot(proj2(:,1),proj2(:,2),'bo');hold off;
figure;imshow(I3);hold on;plot(root_2D_joint3(:,1),root_2D_joint3(:,2),'go');plot(proj3(:,1),proj3(:,2),'bo');hold off;
figure;imshow(I4);hold on;plot(root_2D_joint4(:,1),root_2D_joint4(:,2),'go');plot(proj4(:,1),proj4(:,2),'bo');hold off;


c1 = joints1 - root_2D_joint1;
c2 = joints2 - root_2D_joint2;
c3 = joints3 - root_2D_joint3;
c4 = joints4 - root_2D_joint4;
% figure;imshow(I1);hold on;plot(c1(1),c1(2),'go');hold off;
% figure;imshow(I2);hold on;plot(c2(1),c2(2),'go');hold off;
% figure;imshow(I3);hold on;plot(c3(1),c3(2),'go');hold off;
% figure;imshow(I4);hold on;plot(c4(1),c4(2),'go');hold off;


% 
% [proj1,d1] = ProjectPointRadial(b,cam1(1:9),cam1(10:12),cam1(13:14),cam1(15:16),cam1(17:19),cam1(20:21));
% [proj2,d2] = ProjectPointRadial(b,cam2(1:9),cam2(10:12),cam2(13:14),cam2(15:16),cam2(17:19),cam2(20:21));
% [proj3,d3] = ProjectPointRadial(b,cam3(1:9),cam3(10:12),cam3(13:14),cam3(15:16),cam3(17:19),cam3(20:21));
% [proj4,d4] = ProjectPointRadial(b,cam4(1:9),cam4(10:12),cam4(13:14),cam4(15:16),cam4(17:19),cam4(20:21));

% figure;imshow(I1);hold on;plot(proj1(:,1),proj1(:,2),'ro');hold off;
% figure;imshow(I2);hold on;plot(proj2(:,1),proj2(:,2),'ro');hold off;
% figure;imshow(I3);hold on;plot(proj3(:,1),proj3(:,2),'ro');hold off;
% figure;imshow(I4);hold on;plot(proj4(:,1),proj4(:,2),'ro');hold off;

function Proj = ProjectPointRadial(P, R, T, f, c, k, p)
    R = reshape(R,[3,3]);
    N = size(P,1);
    X = R'*(P'-T'*ones(1,N));
    XX = X(1:2,:)./([1; 1]*X(3,:));
    r2 = XX(1,:).^2 + XX(2,:).^2;
    radial = 1 + dot(repmat(k',[1 N]), [r2; r2.^2; r2.^3], 1);
    tan = p(1)*XX(2,:) + p(2)*XX(1,:);
    XXX = XX.*repmat(radial+tan,[2 1]) + [p(2) p(1)]'*r2;
    Proj = ones(N,1)*f .* XXX' + ones(N,1)*c;
    D = X(3,:);
end