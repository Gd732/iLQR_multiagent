import math
import numpy as np
import matplotlib.pyplot as plt
import yaml

class CAR: 
    PI = math.pi

    RF = 3.0  # distance from rear to vehicle front end
    RB = 1.0  # distance from rear to vehicle back end
    W = 2.0  # width of car
    
def draw_arrow(x, y, theta, L, c):    
    angle = np.deg2rad(30)  
    d = 0.3 * L  
    w = 2  
    
    x_start = x
    y_start = y
    x_end = x + L * np.cos(theta)
    y_end = y + L * np.sin(theta)

    theta_hat_L = theta + CAR.PI - angle
    theta_hat_R = theta + CAR.PI + angle
    x_hat_start = x_end
    x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
    x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)
    y_hat_start = y_end
    y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
    y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)
    
    plt.plot([x_start, x_end], [y_start, y_end], color=c, linewidth=w)
    plt.plot([x_hat_start, x_hat_end_L],
                [y_hat_start, y_hat_end_L], color=c, linewidth=w)
    plt.plot([x_hat_start, x_hat_end_R],
                [y_hat_start, y_hat_end_R], color=c, linewidth=w)

def draw_car(x, y, yaw, color='black'):
    car = np.array([[-CAR.RB, -CAR.RB, CAR.RF, CAR.RF, -CAR.RB],
                    [CAR.W / 2, -CAR.W / 2, -CAR.W / 2, CAR.W / 2, CAR.W / 2]])

    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    car = np.dot(Rot1, car)

    car += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    draw_arrow(x, y, yaw, CAR.W, color)


def get_path_data(data):
    """_summary_

    Args:
        data (dict): dict of data

    Returns:
        X (list): list of x
        Y (list): list of y
        YAW (list): list of yaw
    """
    agents = list(data['schedule'].keys())
    X = []; Y = []; YAW = []
    for agent in agents:
        agent_data = data['schedule'][agent]
        x = []
        y = []
        yaw = []
        for item in agent_data:
            x.append(item['x'])
            y.append(item['y'])
            yaw.append(item['yaw'])
        X.append(x)
        Y.append(y)
        YAW.append(yaw)
    return X, Y, YAW

def get_extension(path_data):
    """
    Args:
        path_data (tuple): path_data consists of X, Y, YAW
    """
    
    for x in path_data:
        max_len = max(len(xi) for xi in x)
        for i, xi in enumerate(x):
            xi_extended = [xi[min(i, len(xi) - 1)] for i in range(max_len)]
            x[i] = xi_extended
    return path_data

def visualize(X, Y, YAW, ox = [], oy = []):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'];
    for k in range(len(X[0])):
        plt.cla()
        plt.plot(ox, oy, "sk")
        
        idx_car = 0
        for x, y, yaw in zip(X, Y, YAW):
            plt.plot(x, y, linewidth=1.5, color=colors[idx_car])
            draw_car(x[-1], y[-1], -yaw[-1], 'dimgray')
            draw_car(x[k], y[k], -yaw[k], colors[idx_car])

            idx_car += 1

        plt.title("CBS Searching Paths")
        plt.axis("equal")
        plt.gca().invert_yaxis()
        plt.pause(0.1)

    plt.show()

def open_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

if __name__ == '__main__':
    X, Y, YAW = get_path_data(open_yaml('output.yaml'))
    X, Y, YAW = get_extension([X, Y, YAW])

    visualize(X, Y, YAW, [], [])

