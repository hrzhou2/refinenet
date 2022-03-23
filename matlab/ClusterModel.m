classdef ClusterModel < handle
    %CLUSTERMODEL
    
    properties
        % for pca
        pca_k_;
        pca_mean_;
        pca_comp_;
        
        % for cluster
        cluster_k_;
        cluster_center_;
        cluster_threshold_;
        with_std_;
    end
    
    methods (Static)
        function errors = compute_errors(N, Ng)
            errors = sum((N - Ng).^2);
            errors = acosd(1 - errors/2);
        end
    end
    
    methods
        function obj = ClusterModel(pca_k, cluster_k, cluster_threshold)
            % pac 
            obj.pca_k_ = pca_k;
            
            % cluster 
            obj.cluster_k_ = cluster_k;
            obj.cluster_threshold_ = cluster_threshold;
            obj.with_std_ = zeros(1, cluster_k);
            obj.with_std_ = logical(obj.with_std_);
        end
        
        function [idx, CL, Nf] = init_cluster(obj, X, Y)
            % Compute cluster info before training
            % pca
            [COEFF, SCORE] = pca(X', 'NumComponents', obj.pca_k_);
            obj.pca_mean_ = mean(X, 2);
            obj.pca_comp_ = COEFF';
            
            % cluster
            nx = size(X, 2);
            opts = statset('UseParallel', false);
            [idx, C] = kmeans(SCORE, obj.cluster_k_, 'Replicates', 5, 'Options', opts);
            CL = cell(1, obj.cluster_k_);
            cluster_nx = zeros(1, obj.cluster_k_);
            for k = 1 : obj.cluster_k_
                CL{k} = (idx == k);
                cluster_nx(k) = sum(CL{k});
            end
            
            % eat the singleton cluster
            [ratio, I] = min(cluster_nx);
            ratio = ratio / nx;
            if ratio < 0.01
                %Warning : unbalanced clustering!
                [~, C] = kmeans(SCORE(idx ~= I, :), obj.cluster_k_, 'Replicates', 5, 'Options', opts);
                [~, idx] = max(bsxfun(@minus,2*C*SCORE',dot(C,C,2)),[],1);
                
                for k = 1 : obj.cluster_k_
                    CL{k} = (idx == k);
                end
            end
            obj.cluster_center_ = C;
            
            % std
            for k = 1 : obj.cluster_k_
                errors = obj.compute_errors(X(1:3, CL{k}), Y(:, CL{k}));
                if mean(errors) < obj.cluster_threshold_
                    obj.with_std_(k) = true;
                    X(:, CL{k}) = mapstd(X(:, CL{k}));
                else
                    obj.with_std_(k) = false;
                end
            end
            Nf = X';
            
        end
        
        function [idx, CL, Nf] = compute_cluster(obj, X)
            % pca
            SCORE = obj.pca_comp_ * bsxfun(@minus, X, obj.pca_mean_);
            
            % cluster
            C = obj.cluster_center_;
            [~,idx] = max(bsxfun(@minus,2*C*SCORE,dot(C,C,2)),[],1);
            
            CL = cell(1,obj.cluster_k_); 
            for i = 1 : obj.cluster_k_
                CL{i} = (idx == i);
                if obj.with_std_(i)
                    X(:,CL{i}) = mapstd(X(:,CL{i}));
                end
            end
            Nf = X';
        end
        
    end
end

