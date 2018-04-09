% use soft-argmax to find the max joint in the heatmap
function joints = heatmapToJoints_Sargmax(heatmapResized, numJoints)
joints = zeros(3, numJoints, 'single');

skew = 9;

int_joints = heatmapTo3DJoints(heatmapResized,numJoints);
int_joints = permute(int_joints,[1,3,2]);
for i = 1:numJoints
    sub_img = heatmapResized(:,:,:, i);
    
    max_loc = int_joints(:,i);
    if skew+max_loc(1)>64 || skew+max_loc(2)>64 ||skew+max_loc(3)>64 
        skew = min([64-max_loc(1),64-max_loc(2),64-max_loc(3)]);
    elseif max_loc(1)-skew<=0||max_loc(2)-skew<=0||max_loc(3)-skew<=0
        skew = min([max_loc(1)-1,max_loc(2)-1,max_loc(3)-1])    
    end
    crop_map = sub_img(max_loc(3)-skew:max_loc(3)+skew, max_loc(2)-skew:max_loc(2)+skew, max_loc(1)-skew:max_loc(1)+skew);
    possible_map = Softmax(crop_map);
%     possible_map = Normalize(crop_map);
    loc_range = (1:2*skew+1);
    map = repmat(loc_range,(2*skew+1)^2,1);
    map = reshape(map,[2*skew+1,2*skew+1,2*skew+1]);
    map2 = permute(map,[3,1,2]);
    map1 = permute(map,[1,3,2]);
    map3 = map;
    
    x_map = (map2-1).*possible_map;
    x_ = sum(x_map(:));
    x = x_+max_loc(3)-(skew+1);
    
    y_map = (map1-1).*possible_map;
    y_ = sum(y_map(:));
    y = y_+max_loc(2)-(skew+1);
    
    z_map = (map3-1).*possible_map;
    z_ = sum(z_map(:));
    z = z_+max_loc(1)-(skew+1);
    
    joints(:, i) = [z y x];
end



% use all the heatmap to compute expectation, but when there are multi
% gaussian distributions, then can not get a right result

% for i = 1:numJoints
%     sub_img = heatmapResized(:,:,:, i);
%     possible_map = sub_img;
% %     possible_map = Softmax(sub_img);
%     
%     loc_range = (1:64);
%     map = repmat(loc_range,64^2,1);
%     map = reshape(map,[64,64,64]);
%     map2 = permute(map,[3,1,2]);
%     map1 = permute(map,[1,3,2]);
%     map3 = map;
%     
%     x_map = map1.*possible_map;
%     x_ = sum(x_map(:));
% %     x = x_/(64^2);
%     x = x_;
%     y_map = map2.*possible_map;
%     y_ = sum(y_map(:));
% %     y = y_/(64^2);
%     y = y_;
%     z_map = map3.*possible_map;
%     z_ = sum(z_map(:));
% %     z = z_/(64^2);
%     z = z_;
%     joints(:, i) = [x y z];
end

function map = Softmax(input)

up = exp(input);
sum_e = sum(up(:));
map = up/sum_e;
xianjiend

function map = Normalize(input)

sum_p = sum(input(:));
map = input/sum_p;

end
