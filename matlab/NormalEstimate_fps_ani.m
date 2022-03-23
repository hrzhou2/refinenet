function [normVectors, range] = NormalEstimate_fps_ani(pts, Ks, rand_num, theta_threshold, max_sizeA)
% multi-scale fitting patch selection
if size(pts, 1) == 3
    pts = pts';
end
npts = size(pts, 1);

kdtree = kdtree_build(pts);
max_k = max(Ks);
min_k = min(Ks);
range = zeros(npts, 1);

%% initialize PCA normals
k_pca = min_k;
TP.sigma_threshold = 0.05;
TP.debug_taproot = 0;
[sigms , normVectors , ~ , ~] = compute_points_sigms_normals_two(pts, k_pca, kdtree, ceil(0.5*k_pca));

feature_threshold = feature_threshold_selection(sigms,TP);

init_feat = find(sigms > feature_threshold);
feature_sigms = sigms(init_feat);
[~, id_sigms] = sort(feature_sigms);
id_feature = init_feat(id_sigms);
nfeatures = length(id_feature)

% bandwidth
bandwidth = compute_bandwidth_points(pts, kdtree, k_pca, 1.5*max_k); % 2*k_bw

%% Plane detection
P(length(Ks)*npts) = struct('k', [], 'pid', [], 'n', []);
S(npts) = struct('patch', [], 'score', []);

for kk=1:length(Ks)
    k_knn = Ks(1,kk);
        
    for i = 1:npts
        knn = kdtree_k_nearest_neighbors(kdtree, pts(i,:), k_knn);
        pts_knn = pts(knn,:);

        % random planes
        knn_feature = intersect(knn, id_feature);
        if isempty(knn_feature), continue; end
        max_sort = 0;
        inner_threshold = 2*bandwidth(1,i);
        tempx = ones(1 , 4) ;
        tempy = ones(k_knn , 1);
        for j = 1 : rand_num
            ran_id = [ 1 , randperm(k_knn - 1 , 3) + 1 ]; 
            points_3 = pts_knn(ran_id , :) ;

            mp = (tempx*points_3)./4;
            points_center = pts_knn - tempy * mp;

            tmp = points_center(ran_id , :); 
            C = tmp'*tmp./size(points_3 , 1); 
            [V,~] = eig(C); 
            fix_normal = V(:,1); 

            dis = (points_center*fix_normal).^2; 
            if dis(1) > inner_threshold^2
                continue;
            end

            dis = exp(-dis./min( inner_threshold.^2 ) ); 

            cur_sort = sum(dis);

            if cur_sort > max_sort
                max_sort = cur_sort;
                n_patch = fix_normal ;
                mp_patch = mp;
            end

        end
        
        % store plane info
        index = (kk-1)*npts+i;
        P(index).k = k_knn;
        P(index).pid = i;
        P(index).n = n_patch;
        P(index).mp = mp_patch;
        
        points_centered = pts_knn - repmat(mp_patch, k_knn, 1);
        dis_plane = (points_centered * n_patch).^2;
        
        for f = 1:length(knn_feature)
            S(knn_feature(f)).patch(end+1) = index;
            bd = bandwidth(1, knn_feature(f))+1e-9;
            temp = dis_plane./((2*bd)^2);
            temp = exp(-temp);
            E = sum(temp);
            S(knn_feature(f)).score(end+1) = (E/k_knn) * exp(-((1 - k_knn/max_k)/3).^2);
        end
    end
end
        
%% Patch selection
for fea = 1:length(id_feature)
    fid = id_feature(fea);
    curS = S(fid);
    % sort by score
    [~, idx] = sort([curS.score], 'descend');
    patch_id = curS.patch(idx);
    curPatch = P(patch_id);
    
    %
    knn_fp = kdtree_k_nearest_neighbors(kdtree, pts(fid,:), min_k);
    A = struct('n', [], 'v', []);
    sizeA = 0;
    for i = 1:length(curPatch)
        % reference point
        n_patch = curPatch(i).n;
        vec_mp = curPatch(i).mp - pts(fid,:);
        dis_plane = vec_mp * n_patch;
        p_ref = pts(fid,:) + n_patch' .* dis_plane;

        % reorient normal
        dis_ref = pts(knn_fp,:) - repmat(p_ref, length(knn_fp), 1);
        if sum(dis_ref * curPatch(i).n) >= 0
            curPatch(i).n = -curPatch(i).n;
        end

        % aniostropic patch selection
        if judge(curPatch(i), A, sizeA, theta_threshold)
            sizeA = sizeA+1;
            A(sizeA).n = curPatch(i).n;
            A(sizeA).v = (p_ref - pts(fid,:))';
            A(sizeA).pid = curPatch(i).pid;
            A(sizeA).k = curPatch(i).k;
        end
        if sizeA >= max_sizeA, break; end
    end
    
    % select from A
    min_sort = 1e9;
    min_id = 0;
    for i = 1:length(A)
        n_ani = A(i).n;
        value_ani = A(i).v' * n_ani;
%         value_ani = norm(A(i).v);
        if value_ani<min_sort
            min_sort = value_ani;
            min_id = i;
        end
    end
    normVectors(fid, :) = A(min_id).n;
end

kdtree_delete(kdtree);

end


function [JUDGE] = judge(curPatch, APatch, sizeA, theta_threshold)
    n_patch_ = curPatch.n;
    JUDGE = true;
    for i = 1:sizeA
        if acosd(dot(n_patch_, APatch(i).n)) <= theta_threshold || ...
                acosd(dot(-n_patch_, APatch(i).n)) <= theta_threshold
            JUDGE = false;
            break;
        end 
    end
    
end
