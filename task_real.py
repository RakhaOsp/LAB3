#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64
import numpy as np
import matplotlib.pyplot as plt

def cubic_coefficients(theta_0, theta_f, t_0, t_f, vel_0=0, vel_f=0):
    A = np.array([[1, t_0, t_0 ** 2, t_0 ** 3],
                  [0, 1, 2 * t_0, 3 * t_0 ** 2],
                  [1, t_f, t_f ** 2, t_f ** 3],
                  [0, 1, 2 * t_f, 3 * t_f ** 2]])
    B = np.array([theta_0, vel_0, theta_f, vel_f])
    coefficients = np.linalg.solve(A, B)
    return coefficients

def cubic_trajectory(coefficients, t):
    a0, a1, a2, a3 = coefficients
    return a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3

def inverse_kinematics_with_orientation(X, Y, theta, L1, L2, L3):
    x = X - L3 * np.cos(theta)
    y = Y - L3 * np.sin(theta)

    cos_q2 = (x ** 2 + y ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1, 1)
    q2 = np.arccos(cos_q2)

    K1 = L1 + L2 * np.cos(q2)
    K2 = L2 * np.sin(q2)
    q1 = np.arctan2(y, x) - np.arctan2(K2, K1)

    q1 = np.arctan2(np.sin(q1), np.cos(q1))
    q3 = theta - q1 - q2
    q3 = np.arctan2(np.sin(q3), np.cos(q3))

    return q1, q2, q3

def forward_kinematics(q1, q2, q3, L1, L2, L3):
    x1 = L1 * np.cos(q1)
    y1 = L1 * np.sin(q1)

    x2 = x1 + L2 * np.cos(q1 + q2)
    y2 = y1 + L2 * np.sin(q1 + q2)

    x3 = x2 + L3 * np.cos(q1 + q2 + q3)
    y3 = y2 + L3 * np.sin(q1 + q2 + q3)

    return (0, 0), (x1, y1), (x2, y2), (x3, y3)

def plot_manipulator(q1, q2, q3, L1, L2, L3):
    joint1, joint2, joint3, end_effector = forward_kinematics(q1, q2, q3, L1, L2, L3)

    plt.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], 'b-', label='Link 1')
    plt.plot([joint2[0], joint3[0]], [joint2[1], joint3[1]], 'g-', label='Link 2')
    plt.plot([joint3[0], end_effector[0]], [joint3[1], end_effector[1]], 'r-', label='Link 3')

    plt.plot(joint1[0], joint1[1], 'bo', label='Joint 1')
    plt.plot(joint2[0], joint2[1], 'go', label='Joint 2')
    plt.plot(joint3[0], joint3[1], 'ro', label='Joint 3')
    plt.plot(end_effector[0], end_effector[1], 'mo', label='End-Effector')

    plt.arrow(end_effector[0], end_effector[1], 0.2 * np.cos(q1 + q2 + q3), 0.2 * np.sin(q1 + q2 + q3),
              head_width=0.1, head_length=0.1, fc='m', ec='m')

    plt.title("Manipulator Configuration")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.pause(0.01)  # Ensure plot is updated

def plot_via_points_trajectory(via_points, L1, L2, L3, t_0, t_f_per_segment):
    rospy.init_node('joint_trajectory_publisher', anonymous=True)

    joint1_pub = rospy.Publisher('motortom2m/command', Float64, queue_size=10)
    joint2_pub = rospy.Publisher('joint4/command', Float64, queue_size=10)
    joint3_pub = rospy.Publisher('end/command', Float64, queue_size=10)

    rate = rospy.Rate(10)  # 10 Hz

    X_traj = []
    Y_traj = []

    for i in range(len(via_points) - 1):
        X_start, Y_start, theta_start = via_points[i]
        X_end, Y_end, theta_end = via_points[i + 1]

        q1_start, q2_start, q3_start = inverse_kinematics_with_orientation(X_start, Y_start, theta_start, L1, L2, L3)
        q1_end, q2_end, q3_end = inverse_kinematics_with_orientation(X_end, Y_end, theta_end, L1, L2, L3)

        coeff_q1 = cubic_coefficients(q1_start, q1_end, t_0, t_f_per_segment)
        coeff_q2 = cubic_coefficients(q2_start, q2_end, t_0, t_f_per_segment)
        coeff_q3 = cubic_coefficients(q3_start, q3_end, t_0, t_f_per_segment)

        segment_times = np.linspace(t_0, t_f_per_segment, 100)

        for t in segment_times:
            q1_pos = cubic_trajectory(coeff_q1, t)
            q2_pos = cubic_trajectory(coeff_q2, t)
            q3_pos = cubic_trajectory(coeff_q3, t)

            joint1_pub.publish(q1_pos)
            joint2_pub.publish(q2_pos)
            joint3_pub.publish(q3_pos)

            plt.clf()  # Clear the previous plot
            plot_manipulator(q1_pos, q2_pos, q3_pos, L1, L2, L3)

            # Store the end-effector position
            _, _, _, (x_end, y_end) = forward_kinematics(q1_pos, q2_pos, q3_pos, L1, L2, L3)
            X_traj.append(x_end)
            Y_traj.append(y_end)

            rospy.loginfo("Published joint angles: q1={}, q2={}, q3={}".format(q1_pos, q2_pos, q3_pos))
            rate.sleep()

    # Plot final end-effector path with via points
    plt.figure()
    plt.plot(X_traj, Y_traj, 'r-', label="End-Effector Path")
    for (X, Y, _) in via_points:
        plt.plot(X, Y, 'bo', label="Via Point")
    plt.title("End-Effector Trajectory with Via Points")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    try:
        # Link lengths for your 3RRR manipulator
        L1 = 0.675 * 2
        L2 = 0.675 * 2
        L3 = 3.167 - 0.675 * 4  # Adjust as per your robot's design

        # Define via points in Cartesian space (X, Y, theta)
        via_points = [
            (3, 0, np.pi / 4),  # Starting point,  # Intermediate via point 1  # Intermediate via po
            (0, 3, np.pi / 4),  # Starting point,  # Intermediate via point 1  # Intermediate via po
        ]

        # Plot the joint-space and Cartesian-space trajectory with via points
        plot_via_points_trajectory(via_points, L1, L2, L3, t_0=0, t_f_per_segment=2)

    except rospy.ROSInterruptException:
        pass

