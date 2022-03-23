classdef GeoNormal < handle
    %GEONORMAL 
    
    properties
        % cluster
        cluster_model_;
        
        % parameter
        sigma_s_;
        sigma_r_;
        self_included_;
        rotate_feature_;
        map_size_;
        use_std;
    end
    
    methods (Static)
        function errors = compute_errors(N, Ng)
            errors = sum((N - Ng).^2);
            errors = acosd(1 - errors/2);
        end
        
        function N = reorient_normal(N, Ng)
            for i = 1:length(N)
                if dot(N(:,i), Ng(:,i)) < 0
                    N(:,i) = -N(:,i);
                end
            end
        end
    end
    
    methods
        function obj = GeoNormal(sigma_s, sigma_r, rotate_feature, self_included, ...
                map_size, pca_k, cluster_k, cluster_threshold)
            
            obj.rotate_feature_ = logical(rotate_feature);
            obj.self_included_ = logical(self_included);
            obj.sigma_s_ = sigma_s;
            obj.sigma_r_ = sigma_r;
            obj.map_size_ = map_size;
            obj.use_std = true;
            
            % cluster model
            obj.cluster_model_ = ClusterModel(pca_k, cluster_k, cluster_threshold);
            
        end

        
        function obj = features_from_normal(obj, path_normal, path_pointcloud, path_output, radius, is_training)
            % Initialize cluster model 
            % Compute normal features from normal

            files = dir(fullfile(path_normal, '*.xyz'));
            nfiles = length(files);
            pnums = zeros(1, nfiles);
            mean_errors = zeros(1, nfiles);
            
            nid = 1;
            for i = 1:nfiles
                
                filename = files(i).name(1:end-4);
                [~, N] = read_xyz(fullfile(path_normal, files(i).name));
                V = read_xyz(fullfile(path_pointcloud, [filename, '.xyz']));
                pts = V';
                pnums(i) = length(pts);
                
                % compute errors
                Ngt = read_xyz(fullfile(path_pointcloud, [filename, '.normals']));
                N = obj.reorient_normal(N, Ngt);
                errors = obj.compute_errors(N, Ngt);
                mean_errors(i) = mean(errors);
                
                % Feature
                [feature, rot_, gt_normals] = calc_feature_NF(obj.sigma_s_, obj.sigma_r_, pts, ...
                    radius, N, obj.rotate_feature_, obj.self_included_, Ngt);
                
                % collect data
                id = nid : nid+pnums(i)-1;
                Nf(id,:) = feature';
                Ng(id,:) = gt_normals';
                Rot(:,:,id) = rot_;
                nid = nid+pnums(i);
            end
            
            % compute cluster
            if is_training
                [idx_cluster, CL, ~] = obj.cluster_model_.init_cluster(Nf', Ng');
            else
                [idx_cluster, CL, ~] = obj.cluster_model_.compute_cluster(Nf');
            end
            
            % data saved to file
            writeNPY(single(Nf), fullfile(path_output, 'Nf.npy'));
            writeNPY(single(Ng), fullfile(path_output, 'Ng.npy'));
            % to row-major
            Rot = permute(Rot, [3,1,2]);
            writeNPY(single(Rot), fullfile(path_output, 'Rot.npy'));

            % Save Info
            writeNPY(idx_cluster, fullfile(path_output, 'idx_cluster.npy'));
            
            % Save Error
            fid = fopen(fullfile(path_output, 'ErrorInfo.txt'), 'w');
            fprintf(fid, '%d  %d  %f\n', nfiles, sum(pnums), sum(pnums .* mean_errors) / sum(pnums));

            for i = 1:length(files)
                fprintf(fid, '%s  %d  %f\n', files(i).name, pnums(i), mean_errors(i));
            end
            fclose(fid);
            
            % print cluster info
            for k = 1 : obj.cluster_model_.cluster_k_
                error = obj.compute_errors(Nf(CL{k}, 1:3)', Ng(CL{k}, :)');
                fprintf('%d %f\n', sum(CL{k}), mean(error));
            end

        end
        
    end
end

