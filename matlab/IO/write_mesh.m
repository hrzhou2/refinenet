function write_mesh(filename, V, F)
% write_mesh - read data to OFF, OBJ file.


[~, ~, EXT] = fileparts(filename);

switch lower(EXT)
    case '.off'
        write_off(filename, V, F);
    case '.obj'
        write_obj(filename, V, F);
    otherwise
        error('read_mesh : Unknown extension.');
end