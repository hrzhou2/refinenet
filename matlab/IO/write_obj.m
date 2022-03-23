function [ output_args ] = write_obj(filename, V, N)
%WRITE_OBJ �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
fid=fopen(filename,'wt');
if(fid==-1)
    error('can''t open the file');
    return;
end
%header
fprintf(fid,'# %d vertices\n', length(V));
%write the points & normals
fprintf(fid,'v %f %f %f\n', V);
fprintf(fid,'vn %f %f %f\n', N);
fclose(fid);

end

