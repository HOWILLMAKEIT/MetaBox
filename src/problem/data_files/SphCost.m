function [J1, J2, J3, J4]=SphCost(sol,model)
% This function calculates the path cost of a given solution "sol" over a 
% certain setting (model).
% sol must be a 3*dimenison-by-1 column vector of spherical coordinates
    
    J_pen = model.J_pen;
    n = model.n;
    H = model.H;


    % Input solution
    [x,y,z]=SphericalToCart_vec(sol,model); %transforming the solution to the cartesian coordinates system
    
    % Start location
    xs=model.start(1);
    ys=model.start(2);
    zs=model.start(3);
    
    % Final location
    xf=model.end(1);
    yf=model.end(2);
    zf=model.end(3);

    x_all = [xs x xf];
    y_all = [ys y yf];
    z_all = [zs z zf];
    
    N = size(x_all,2); % Full path length
    
    % Altitude wrt sea level = z_relative + ground_level
    z_abs = zeros(1,N);
    for i = 1:N
        z_abs(i) = z_all(i) + H(round(y_all(i)),round(x_all(i)));
    end
    %============================================
    % J1 - Cost for path length    
    J1 = 0;
    for i = 1:N-1
        diff = [x_all(i+1) - x_all(i);y_all(i+1) - y_all(i);z_abs(i+1) - z_abs(i)];
        J1 = J1 + norm(diff);
    end

    %==============================================
    % J2 - threats/obstacles Cost   

    % Threats/Obstacles
    threats = model.threats;
    threat_num = size(threats,1);

    % Checking if UAV passes through a threat
    J2 = 0;
    for i = 1:threat_num
        threat = threats(i,:);
        threat_x = threat(1);
        threat_y = threat(2);
        threat_radius = threat(4);
        for j = 1:N-1
            % Distance between projected line segment and threat origin
            dist = DistP2S([threat_x threat_y],[x_all(j) y_all(j)],[x_all(j+1) y_all(j+1)]);
            if dist > (threat_radius + model.drone_size + model.danger_dist) % No collision
                threat_cost = 0;
            elseif dist < (threat_radius + model.drone_size)  % Collision
                threat_cost = J_pen;
            else  % Dangerous Zone
                threat_cost = (threat_radius + model.drone_size + model.danger_dist) - dist;
            end
            J2 = J2 + threat_cost;
        end
    end

    %==============================================
    % J3 - Altitude cost
    % Note: In this calculation, z, zmin & zmax are heights with respect to the ground
    zmax = model.zmax;
    zmin = model.zmin;
    J3 = 0;
    for i=1:n        
        if z(i) < 0   % crash into ground
            J3_node = J_pen;
        else
            J3_node = abs(z(i) - (zmax + zmin)/2); 
        end
        
        J3 = J3 + J3_node;
    end
    
    %==============================================
    % J4 - Smooth cost
    J4 = 0;
    turning_max = 45;
    climb_max = 45;
    for i = 1:N-2
        
        % Projection of line segments to Oxy ~ (x,y,0)
        for j = i:-1:1
             segment1_proj = [x_all(j+1); y_all(j+1); 0] - [x_all(j); y_all(j); 0];
             if nnz(segment1_proj) ~= 0
                 break;
             end
        end

        for j = i:N-2
            segment2_proj = [x_all(j+2); y_all(j+2); 0] - [x_all(j+1); y_all(j+1); 0];
             if nnz(segment2_proj) ~= 0
                 break;
             end
        end
     
        climb_angle1 = atan2d(z_abs(i+1) - z_abs(i),norm(segment1_proj));
        climb_angle2 = atan2d(z_abs(i+2) - z_abs(i+1),norm(segment2_proj));
       
        turning_angle = atan2d(norm(cross(segment1_proj,segment2_proj)),dot(segment1_proj,segment2_proj));
       
        if abs(turning_angle) > turning_max
            J4 = J4 + abs(turning_angle);
        end
        if abs(climb_angle2 - climb_angle1) > climb_max
            J4 = J4 + abs(climb_angle2 - climb_angle1);
        end
       
    end
end