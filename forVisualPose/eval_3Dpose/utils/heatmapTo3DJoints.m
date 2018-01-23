% Find joints in heatmap (== max locations in heatmap)
function joints = heatmapTo3DJoints(heatmapResized, numJoints)
joints = zeros(3, numJoints, 'single');

for i = 1:numJoints
    sub_img = heatmapResized(:,:,:, i);
    vec = sub_img(:);
    [val,idx] = max(vec);
    [z,y,x] = ind2sub(size(sub_img), idx);
    joints(:, i) = [x y z];
end