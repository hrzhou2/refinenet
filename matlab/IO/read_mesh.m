function [V, F] = read_mesh(filename)

% read_mesh - read data from OFF, OBJ files
%
%   [vertex,face] = read_mesh(filename);
%   [vertex,face] = read_mesh;      % open a dialog box
%
%   'V' : '3 x nv' matrix.
%   'F' : '3 x nf' matrix


if nargin==0
    [name, pathname] = uigetfile({'*.obj;*.off','*obj,*.off'},'Open file');
    filename = [pathname, name];
end

[~, ~, EXT] = fileparts(filename);

switch lower(EXT)
    case '.off'
        [V, F] = read_off(filename);
    case '.obj'
        [V, F] = read_obj(filename);
    otherwise
        error('read_mesh : Unknown extension.');
end