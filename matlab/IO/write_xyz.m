function [] = write_xyz(filename, V, N)
fid=fopen(filename,'wt');
if(fid==-1)
    error('can''t open the file');
    return;
end

if exist('N', 'var')
    V = [V;N];
    fprintf(fid, '%f %f %f %f %f %f\n', V);
else
    fprintf(fid, '%f %f %f\n', V);
end

fclose(fid);

end