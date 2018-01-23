function errorH36M(S,Sgt)

    % compute error
    E3D = computeError(S,Sgt);
    % print per action results
    printResults(E3D)
    
end

function dist = computeError(S,Sgt)

    dist = zeros(size(S,1),1);
    for i = 1:size(S,1)
        S1 = squeeze(S(i,:,:)); 
        S2 = squeeze(Sgt(i,:,:));
        % root alignment
        S1 = S1 - repmat(S1(:,1),1,size(S1,2));
        S2 = S2 - repmat(S2(:,1),1,size(S2,2));
        % mean per joint 3D error
        dist(i,1) = mean(sqrt(sum((S1-S2).^2,1)));
    end

end

function printResults(E3D)

    subject_set = {'S9','S11'};
    motion_set = {'Directions','Discussion','Eating','Greeting','Phoning',...
        'Posing','Purchases','Sitting','SittingDown',...
       'Photo', 'Smoking','Waiting','WalkDog','Walking','WalkTogether'};

    % per action error
    for i = 1:15
        E3D_all{i,1} = [];
    end
    for i = [1:2699,39011:40562]
        E3D_all{1} =[E3D_all{1}; E3D(i)];
    end
    for i = [2700:8005,40563:42760]
        E3D_all{2} =[E3D_all{2}; E3D(i)];
    end
    for i = [8006:10691,42761:44963]
        E3D_all{3} =[E3D_all{3}; E3D(i)];
    end
    for i = [10692:12138,44964:46771]
        E3D_all{4} =[E3D_all{4}; E3D(i)];
    end
    for i = [12139:15457,46772:50263]
        E3D_all{5} =[E3D_all{5}; E3D(i)];
    end
    for i = [15458:17421,50264:51670]
        E3D_all{6} =[E3D_all{6}; E3D(i)];
    end
    for i = [17422:18950,51671:52710]
        E3D_all{7} =[E3D_all{7}; E3D(i)];
    end
    for i = [18951:21912,52711:54889]
        E3D_all{8} =[E3D_all{8}; E3D(i)];
    end
    for i = [21913:23466,54890:56893]
        E3D_all{9} =[E3D_all{9}; E3D(i)];
    end
    for i = [23467:27800,56894:59303]
        E3D_all{10} =[E3D_all{10}; E3D(i)];
    end
    for i = [27801:30146,59304:61293]
        E3D_all{11} =[E3D_all{11}; E3D(i)];
    end
    for i = [30147:33458,61294:63555]
        E3D_all{12} =[E3D_all{12}; E3D(i)];
    end
    for i = [33459:35070,63556:65176]
        E3D_all{13} =[E3D_all{13}; E3D(i)];
    end
    for i = [35071:37307,65177:66611]
        E3D_all{14} =[E3D_all{14}; E3D(i)];
    end
    for i = [37308:39010,66612:67971]
        E3D_all{15} =[E3D_all{15}; E3D(i)];
    end
    
    
    
%     for img_i = 1:numel(imgname)
%         motion = char(regexp(imgname{img_i},'_[a-zA-Z]+','match'));
%         motion = motion(2:end);
%         for motion_i = 1:numel(motion_set)
%             if strcmp(motion, motion_set{motion_i})
%                 E3D_all{motion_i} =[E3D_all{motion_i}; E3D(img_i)];
%                 break
%             end
%         end
%     end

    % print results
    fid = fopen('results.txt','a+');
    fprintf(fid,'\n');
    fprintf(fid,datestr(now));
    fprintf(fid,'  %s ',subject_set{:});
    fprintf(fid,'\n');
    fprintf(fid,'%12s','Motion ');
    for i = 1:length(motion_set)
        fprintf(fid,'& %12s ',motion_set{i});
    end
    fprintf(fid,'& %12s','Average');
    fprintf(fid,'\\\\\n');
    fprintf(fid,'%12s','Final ');
    for i = 1:length(E3D_all)
        if ~isempty(E3D_all{i})
            fprintf(fid,'& %12s ',sprintf('%.2f',mean(E3D_all{i})));
        else
            fprintf(fid,'& %12s ','NaN');
        end
    end
    fprintf(fid,'& %12s ',sprintf('%.2f',mean(mean(cell2mat(E3D_all)))));
    fprintf(fid,'\\\\\n');
    fclose(fid);
    
end