function []=write_off(filename,V)

%   write_off(filename, vertex);
%
%   vertex must be of size [3,n]
%
fid=fopen(filename,'wt');
if(fid==-1)
    error('can''t open the file');
    return;
end
%header
fprintf(fid,'OFF\n');
fprintf(fid, '%d 0\n', length(V));
%write the points & normals
fprintf(fid,'%f %f %f\n',V);
fclose(fid);

end