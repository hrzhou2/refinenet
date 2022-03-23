function [ feature, Rot ] = NormalFilter_MS_ball_abs( sigma_s, sigma_r, pts, ...
                 normals, radius_rate, rotate_feature, self_included)
% use ball neighbor, radius = diagonal*radius_rate, 
% len=0.5*radius, sigma_s = ()*len
% abs to reorient neighboring normals, now use one_minus_cos for normal_dis

% setting
npts = size(pts, 1);
ns = length(sigma_s);
nr = length(sigma_r);
n_input = ns*nr*3;
if self_included, n_input = n_input+3; end
feature = zeros(n_input, npts);
Rot = zeros(3, 3, npts);

bbox = max(pts, [], 1) - min(pts, [], 1);
diagonal = sqrt(sum(bbox.^2));
radius = diagonal*radius_rate;
len = radius*0.5;

% bilateral filtering
kdtree = kdtree_build(pts);
cnt_knn = 0;

for i = 1:npts
    R = zeros(3,3);

    [knn, dis] = kdtree_ball_query(kdtree, pts(i,:), radius);
    dis2 = dis.^2;
    k_knn = length(knn);
    cnt_knn = cnt_knn + k_knn;
    
    normals_knn = normals(knn,:);
    cos_theta = dot(normals_knn, repmat(normals(i,:), k_knn, 1), 2);
    cos_theta = abs(cos_theta);
    one_minus_cos2 = (1-cos_theta)*2;

    nid = 1;
    if self_included
        feature((nid*3-2):(nid*3), i) = normals(i,:)';
        nid = nid+1;
        R = R + normals(i,:)'*normals(i,:);
    end
    for ss = 1:ns
        for rr = 1:nr
            s = (sigma_s(ss) * len).^2;
            r = 2 * sigma_r(rr).^2;
            
            E = exp(-dis2./s) .* exp(-one_minus_cos2./r );
            n_sum = sum(E .* normals_knn, 1);
            n_sum = (n_sum/norm(n_sum))';

            feature((nid*3-2):(nid*3),i) = n_sum;
            R = R+n_sum*n_sum';
            nid = nid+1;
        end
    end
    if rotate_feature
        [V, D] = eig(R);
        [~,ind] = sort(diag(D));
        Vs = V(:,ind);
        Rot(:,:,i) = Vs';
        for j = 1:n_input/3
            feature((j*3-2):(j*3), i) = Vs' * feature((j*3-2):(j*3), i);
        end
    end
end
cnt_knn = cnt_knn/npts;
kdtree_delete(kdtree);
end

