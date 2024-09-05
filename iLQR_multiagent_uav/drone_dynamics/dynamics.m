syms p_x p_y p_z pdot_x pdot_y pdot_z theta phi psi thetadot phidot psidot real
syms T1 T2 T3 T4 m g L k b Ix Iy Iz kd dt real
syms p_x_next p_y_next p_z_next pdot_x_next pdot_y_next pdot_z_next theta_next phi_next psi_next thetadot_next phidot_next psidot_next real
x = [p_x; p_y; p_z; pdot_x; pdot_y; pdot_z; theta; phi; psi; thetadot; phidot; psidot];
u = [T1; T2; T3; T4];
I = diag([Ix, Iy, Iz]);
u_act = u.^2 + [sqrt(9.81/4),sqrt(9.81/4),sqrt(9.81/4),sqrt(9.81/4)]'.^2;
%%
p = [p_x, p_y, p_z]';
pdot = [pdot_x, pdot_y, pdot_z]';
theta = [theta, phi, psi]';
thetadot = [thetadot, phidot, psidot]';
p_next = [p_x_next, p_y_next, p_z_next]';
pdot_next = [pdot_x_next, pdot_y_next, pdot_z_next]';
theta_next = [theta_next, phi_next, psi_next]';
thetadot_next = [thetadot_next, phidot_next, psidot_next]';
%%
s0=sin(theta(1));c0=cos(theta(1));
s1=sin(theta(2));c1=cos(theta(2));

R = [1, 0, -s1; 0, c0, c1*s0; 0, -s0, c1*c0];
R_inv = [1, s0*s1/c1, c0*s1/c1; 0, c0, -s0; 0, s0/c1, c0/c1];
%%
omega = R*thetadot;

T = R*[0; 0; sum(u_act)*k/m];
a = [0; 0; -9.81] + T - kd*pdot;
%%
tau=[L*k*(u_act(1)-u_act(2));
    L*k*(u_act(2)-u_act(4));
    b*(u_act(1)-u_act(2)+u_act(3)-u_act(4))];
omegadot = inv(I) * (tau - cross(omega, I * omega));
omega = omega + dt*omegadot;
%%
thetadot_next = R_inv * omega;
theta_next = theta + dt*thetadot_next;
pdot_next = pdot + dt*a;
p_next = p + dt*pdot;

xnext = [p_next; pdot_next; theta_next; thetadot_next];
%%
A = jacobian(xnext,x);
B = jacobian(xnext,u);