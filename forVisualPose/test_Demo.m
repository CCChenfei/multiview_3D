
clear;
opt.visualise = false;
opt.useGPU = true;
opt.dims = [256 256];
opt.numJoints = 17;
joints_out = zeros(2,17,67971);
numimages = 67971;
for ind = 1 : numimages
    img = imread(['/home/chenf/Documents/pose_estimation/data/multiviewH36m17j/test/c4/im',num2str(ind),'.jpg']);
    imgPose = prepareImagePose(img);
    
%     load(['/home/chenf/PycharmProjects/hourglasstf/joints_multiview/joint',num2str(ind),'.mat']);
%     fileinfo = hdf5info('/home/chenf/PycharmProjects/multiview_3D/out_heatmap.h5')
    joint = h5read('//home/chenf/PycharmProjects/multiview_3D/out_heatmap.h5',['/out',num2str(ind)]);
%     load(['/home/chenf/PycharmProjects/multiview_3D/output2.mat']);
%     output = output(1,1,:,:,:);
%     output = reshape(output,[64,64,17]);
    output = joint(:,:,:,1,4);
    output = permute(output,[3,2,1]);
    [joints,heatmaps] = processHeatmap(output, opt);
    joints_out(:,:,ind) = joints;
    
    if opt.visualise
        heatmapVis = getConfidenceImage(heatmaps,img);
        figure(2),imshow(heatmapVis);
        
        figure(1),imshow(uint8(img));
        hold on
        
        plotSkeleton_14(joints,[],[]);
        hold off
    end
     if opt.visualise; waitforbuttonpress; end
        
end
save('joints4.mat','joints_out');