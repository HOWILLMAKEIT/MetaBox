% Convert the solution from 3*dim-by-1 sloution spherical space vector to Cartesian coordinates
% The input here is a column vector 
function [x,y,z] = SphericalToCart_vec(sol,model)
    % Start location
    xs = model.start(1);
    ys = model.start(2);
    zs = model.start(3);
    % getting r,phi,psi
    for i=1:size(sol,1)
        if mod(i,3)==1
        r(fix(i/3)+1)=sol(i);
        end        
        if mod(i,3)==0
        phi(fix(i/3))=sol(i);
        end
        if mod(i,3)==2
        psi(fix(i/3)+1)=sol(i);
        end
    end
    % First Cartesian coordinate
    x(1) = xs + r(1)*cos(psi(1))*sin(phi(1));
    
    % Check limits
    if x(1) > model.xmax
        x(1) = model.xmax;
    end
    if x(1) < model.xmin
        x(1) = model.xmin;
    end 
    
     y(1) = ys + r(1)*cos(psi(1))*cos(phi(1));
    if y(1) > model.ymax
        y(1) = model.ymax;
    end
    if y(1) < model.ymin
        y(1) = model.ymin;
    end
    
     z(1) = zs + r(1)*sin(psi(1));
    if z(1) > model.zmax
        z(1) = model.zmax;
    end
    if z(1) < model.zmin
        z(1) = model.zmin;
    end 
    
    % Next Cartesian coordinates
    for i = 2:model.n
        x(i) = x(i-1) + r(i)*cos(psi(i))*sin(phi(i));
        if x(i) > model.xmax
            x(i) = model.xmax;
        end
        if x(i) < model.xmin
            x(i) = model.xmin;
        end 

        y(i) = y(i-1) + r(i)*cos(psi(i))*cos(phi(i));
        if y(i) > model.ymax
            y(i) = model.ymax;
        end
        if y(i) < model.ymin
            y(i) = model.ymin;
        end

        z(i) = z(i-1) + r(i)*sin(psi(i));
        if z(i) > model.zmax
            z(i) = model.zmax;
        end
        if z(i) < model.zmin
            z(i) = model.zmin;
        end 
    end
    
end