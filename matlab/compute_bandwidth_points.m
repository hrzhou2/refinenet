function [bandwidth] = compute_bandwidth_points(pts, kdtree, k_knn, knn_select)
%COMPUTE_BANDWIDTH_POINTS 

npts = size(pts, 1);
voting = repmat([1e9], 1, npts);
err = zeros(1,npts);

for i = 1:npts
    knn = kdtree_k_nearest_neighbors(kdtree, pts(i,:), k_knn);

   % apply PCA on patch
    pts_knn = pts(knn,:);
    mp = mean(pts_knn);
    tmp = pts_knn-repmat(mp,k_knn,1);
    C = tmp'*tmp./k_knn;
    [V, D] = eig(C);
    n_patch = V(:,1);
    n_patch = n_patch/norm(n_patch);

   % compute the consistency of patch
    d = abs( (pts_knn - repmat(mp, k_knn, 1)) * n_patch );
    E = sum(d) + 1e-9;
    
    mean_threshold = E/k_knn; % diff
    for j=1:k_knn
        dis = abs((pts_knn(j,:) - pts(i,:)) * n_patch);
        t = exp((dis/(2*mean_threshold))^2);
        if E*t<voting(knn(j,1))
            voting(knn(j,1)) = E*t;
            err(knn(j,1)) = d(j);
        end
    end
end

bandwidth = zeros(1, npts);
for i = 1:npts
    knn = kdtree_k_nearest_neighbors(kdtree, pts(i,:), knn_select); 
    bandwidth(i) = mean(err(knn));
end

end

