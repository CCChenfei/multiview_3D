
clear;
opt.visualise = true;
opt.useGPU = true;
opt.dims = [256 256];
opt.numJoints = 14;

numimages = 28961;
for ind = 1 : numimages
    img = imread(['/home/chenf/Documents/pose_estimation/data/Human3.6Mall/test/c1/im',num2str(ind),'.jpg']);
    imgPose = prepareImagePose(img);
    
    load(['/home/chenf/PycharmProjects/hourglasstf/joints/joint',num2str(ind),'.mat']);
    output = joint;
    [joints,heatmaps] = processHeatmap(output, opt);
    
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
