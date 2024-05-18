import argparse

def add_arguments(parser):
    # ___________________ Planning Parameters ___________________ #
    parser.add_argument('--number_of_local_wpts', type= int, default=20, help='Number of local waypoints')
    parser.add_argument('--poly_order', type= int, default=5, help='Order of the polynomial to fit on')
    parser.add_argument('--desired_speed', type= float, default=5.0, help='Desired Speed')

    # ___________________ iLQR Parameters ___________________ #
    parser.add_argument('--timestep', type=float, default=0.1, help='Timestep at which forward and backward pass are done by iLQR')
    parser.add_argument('--horizon', type=int, default=20, help='Planning horizon for iLQR in num of steps (T=horizon*timesteps)')
    parser.add_argument('--tol', type=float, default=1e-4, help='iLQR tolerance parameter for convergence')
    parser.add_argument('--max_iters', type=int, default=20, help='Total number of iterations for iLQR')
    parser.add_argument('--num_states', type=int, default=4, help='Number of states in the model')
    parser.add_argument('--num_ctrls', type=int, default=2, help='Number of control inputs in the model')

    # ___________________ Cost Parameters ___________________ #
    parser.add_argument('--w_acc', type=float, default=1.00, help="Acceleration cost")
    parser.add_argument('--w_yawrate', type=float, default=5.00, help="Yaw rate cost")
    parser.add_argument('--w_pos', type=float, default=2.0, help="Path deviation cost")
    parser.add_argument('--w_vel', type=float, default=0.50, help="Velocity cost")
    parser.add_argument('--q1_acc', type=float, default=1.0, help="Barrier function q1, acc")
    parser.add_argument('--q2_acc', type=float, default=1.0, help="Barrier function q2, acc")
    parser.add_argument('--q1_yawrate', type=float, default=1.00, help="Barrier function q1, yawrate")
    parser.add_argument('--q2_yawrate', type=float, default=1.00, help="Barrier function q2, yawrate")
    parser.add_argument('--q1_front', type=float, default=2.75, help="Barrier function q1, obs with agent front")
    parser.add_argument('--q2_front', type=float, default=2.75, help="Barrier function q2, obs with agent front")
    parser.add_argument('--q1_rear', type=float, default=2.5, help="Barrier function q1, obs with agent rear")
    parser.add_argument('--q2_rear', type=float, default=2.5, help="Barrier function q2, obs with agent rear")

    # ___________________ Constraint Parameters ___________________ #
    parser.add_argument('--acc_limits', nargs="*", type=float, default=[-5.5, 2.0], help="Acceleration limits for the agent vehicle (min,max)")
    parser.add_argument('--steer_angle_limits', nargs="*", type=float, default=[-1.0, 1.0], help="Steering Angle limits (rads) for the agent vehicle (min,max)")

    # ___________________ Ego Vehicle Parameters ___________________ #
    parser.add_argument('--wheelbase', type=float, default=3.0, help="Ego Vehicle's wheelbase")
    parser.add_argument('--max_speed', type=float, default=30.0, help="Ego Vehicle's max speed")
    parser.add_argument('--steering_control_limits', nargs="*", type=float, default=[-1.0, 1.0], help="Steering control input limits (min,max)")
    parser.add_argument('--throttle_control_limits', nargs="*", type=float, default=[-1.0, 1.0], help="Throttle control input limits (min,max)")

    # ___________________ Obstacle Parameters ___________________ #
    parser.add_argument('--t_safe', type=float, default=0.1, help="Time safety headway")
    parser.add_argument('--s_safe_a', type=float, default=10.0, help="safety margin longitudinal")
    parser.add_argument('--s_safe_b', type=float, default=4.0, help="safety margin lateral")
    parser.add_argument('--agent_rad', type=float, default=2, help="Ego Vehicle's radius")
    parser.add_argument('--agent_lf', type=float, default=1.47, help="Distance to front tire")
    parser.add_argument('--agent_lr', type=float, default=1.47, help="Distance to rear tire")