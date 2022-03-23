function [V,N]=read_xyz(filename)

fid = fopen(filename,'r');
if( fid==-1 )
    error('Can''t open the file.');
    return;
end

str = fgetl(fid);
tmp = sscanf(str,'%f %f %f');

frewind(fid);

if length(tmp) < 4 % only x y z
    [A,cnt] = fscanf(fid,'%f %f %f');
    A = reshape(A, 3, cnt/3);
    V = A;
    N = [];
else % x y z nx ny nz
    [input, cnt] = fscanf(fid, '%f');
    input = reshape(input, 6, cnt/6);
    V = input(1:3, :);
    N = input(4:6, :);
end

fclose(fid);

end