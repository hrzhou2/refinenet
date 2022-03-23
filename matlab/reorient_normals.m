function [N] = reorient_normals(N, Ng)
% reorient according to gt normals
% do not consider orientation in training
    for i = 1:length(N)
        if dot(N(:,i), Ng(:,i)) < 0
            N(:,i) = -N(:,i);
        end
    end
end

