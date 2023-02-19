import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid , Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np
import heapq
import math
import scipy.interpolate as si
import sys
import random
import yaml

with open("src/autonomous_exploration/config/params.yaml", 'r') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

lookahead_distance = params["lookahead_distance"]
speed = params["speed"]
expansion_size = params["expansion_size"]
target_error = params["target_error"]
robot_r = params["robot_r"]


def euler_from_quaternion(x,y,z,w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return yaw_z

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def astar(array, start, goal):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data = data + [start]
            data = data[::-1]
            return data
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    # If no path to goal was found, return closest path to goal
    if goal not in came_from:
        closest_node = None
        closest_dist = float('inf')
        for node in close_set:
            dist = heuristic(node, goal)
            if dist < closest_dist:
                closest_node = node
                closest_dist = dist
        if closest_node is not None:
            data = []
            while closest_node in came_from:
                data.append(closest_node)
                closest_node = came_from[closest_node]
            data = data + [start]
            data = data[::-1]
            return data
    return False

def bspline_planning(array, sn):
    try:
        array = np.array(array)
        x = array[:, 0]
        y = array[:, 1]
        N = 2
        t = range(len(x))
        x_tup = si.splrep(t, x, k=N)
        y_tup = si.splrep(t, y, k=N)

        x_list = list(x_tup)
        xl = x.tolist()
        x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

        y_list = list(y_tup)
        yl = y.tolist()
        y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

        ipl_t = np.linspace(0.0, len(x) - 1, sn)
        rx = si.splev(ipl_t, x_list)
        ry = si.splev(ipl_t, y_list)
        path = [(rx[i],ry[i]) for i in range(len(rx))]
    except:
        path = array
    return path

def pure_pursuit(current_x, current_y, current_heading, path,lookahead_distance,index):
    closest_point = None
    v = speed
    for i in range(index,len(path)):
        x = path[i][0]
        y = path[i][1]
        distance = math.hypot(current_x - x, current_y - y)
        if lookahead_distance < distance:
            closest_point = (x, y)
            index = i
            break
    if closest_point is not None:
        target_heading = math.atan2(closest_point[1] - current_y, closest_point[0] - current_x)
        desired_steering_angle = target_heading - current_heading
    else:
        target_heading = math.atan2(path[-1][1] - current_y, path[-1][0] - current_x)
        desired_steering_angle = target_heading - current_heading
        index = len(path)-1
    if desired_steering_angle > math.pi:
        desired_steering_angle -= 2 * math.pi
    elif desired_steering_angle < -math.pi:
        desired_steering_angle += 2 * math.pi
    if desired_steering_angle > math.pi/6 or desired_steering_angle < -math.pi/6:
        sign = 1 if desired_steering_angle > 0 else -1
        desired_steering_angle = sign * math.pi/4
        v = 0.0
    return v,desired_steering_angle,index

def frontierB(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0.0:
                if i > 0 and matrix[i-1][j] < 0:
                    matrix[i][j] = 2
                elif i < len(matrix)-1 and matrix[i+1][j] < 0:
                    matrix[i][j] = 2
                elif j > 0 and matrix[i][j-1] < 0:
                    matrix[i][j] = 2
                elif j < len(matrix[i])-1 and matrix[i][j+1] < 0:
                    matrix[i][j] = 2
    return matrix

def assign_groups(matrix):
    group = 1
    groups = {}
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 2:
                group = dfs(matrix, i, j, group, groups)
    return matrix, groups

def dfs(matrix, i, j, group, groups):
    if i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]):
        return group
    if matrix[i][j] != 2:
        return group
    if group in groups:
        groups[group].append((i, j))
    else:
        groups[group] = [(i, j)]
    matrix[i][j] = 0
    dfs(matrix, i + 1, j, group, groups)
    dfs(matrix, i - 1, j, group, groups)
    dfs(matrix, i, j + 1, group, groups)
    dfs(matrix, i, j - 1, group, groups)
    dfs(matrix, i + 1, j + 1, group, groups) # sağ alt çapraz
    dfs(matrix, i - 1, j - 1, group, groups) # sol üst çapraz
    dfs(matrix, i - 1, j + 1, group, groups) # sağ üst çapraz
    dfs(matrix, i + 1, j - 1, group, groups) # sol alt çapraz
    return group + 1

def deleteGroups(groups):
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    top_five_groups = sorted_groups[:5]
    if len(sorted_groups) > 5:
        for group in sorted_groups[5:]:
            if group not in top_five_groups:
                del groups[group[0]] 
    return groups

def calculate_centroid(x_coords, y_coords):
    n = len(x_coords)
    sum_x = sum(x_coords)
    sum_y = sum(y_coords)
    mean_x = sum_x / n
    mean_y = sum_y / n
    centroid = (int(mean_x), int(mean_y))
    return centroid

def findClosestGroup(matrix,groups, current,resolution,originX,originY):
    min = 1000
    target = None
    for value in groups.items():
        middle = calculate_centroid([p[0] for p in value[1]],[p[1] for p in value[1]])
        path = astar(matrix, current, middle)
        path = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
        total_distance = pathLength(path)
        if total_distance < min and total_distance > target_error*2:
            target = middle
            min = total_distance
        if target == None:
            target = value[1][random.randint(0,len(value[1])-1)]
    return target[0],target[1]

def pathLength(path):
    for i in range(len(path)):
        path[i] = (path[i][0],path[i][1])
        points = np.array(path)
    differences = np.diff(points, axis=0)
    distances = np.hypot(differences[:,0], differences[:,1])
    total_distance = np.sum(distances)
    return total_distance

def costmap(data,width,height,resolution):
    data = np.array(data).reshape(height,width)
    wall = np.where(data == 100)
    for i in range(-expansion_size,expansion_size+1):
        for j in range(-expansion_size,expansion_size+1):
            if i  == 0 and j == 0:
                continue
            x = wall[0]+i
            y = wall[1]+j
            x = np.clip(x,0,height-1)
            y = np.clip(y,0,width-1)
            data[x,y] = 100
    data = data*resolution
    return data

def exploration(data,width,height,resolution,column,row,originX,originY):
        data = costmap(data,width,height,resolution)
        data[row][column] = 0 #Robot Anlık Konum
        data[data > 5] = 1 # 0 olanlar gidilebilir yer, 100 olanlar kesin engel
        data = frontierB(data)
        data,groups = assign_groups(data)
        groups = deleteGroups(groups)
        if len(groups) == 0:
            print("[BILGI] KESIF TAMAMLANDI")
            sys.exit()
        data[data < 0] = 1 #-0.05 olanlar bilinmeyen yer
        #data icerigi 0: gidilebilir, 1: engel
        rowH,columnH = findClosestGroup(data,groups,(row,column),resolution,originX,originY)
        path = astar(data,(row,column),(rowH,columnH))
        path = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
        path = bspline_planning(path,len(path)*5)
        return path

def localControl(scan):
    v = None
    w = None
    for i in range(60):
        if scan[i] < robot_r:
            v = 0.2
            w = -math.pi/4 
            break
    if v == None:
        for i in range(300,360):
            if scan[i] < robot_r:
                v = 0.2
                w = math.pi/4
                break
    return v,w


class navigationControl(Node):
    def __init__(self):
        super().__init__('Exploration')
        self.subscription = self.create_subscription(OccupancyGrid,'map',self.listener_callback,10)
        self.subscription = self.create_subscription(Odometry,'odom',self.info_callback,10)
        self.subscription = self.create_subscription(LaserScan,'scan',self.scan_callback,10)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.01
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.flag = 1
        print("[BILGI] KESİF MODU AKTİF")
        print("[BILGI] YENI HEDEF BELİRLENİYOR")

    def scan_callback(self,msg):
        self.scan = msg.ranges

    def listener_callback(self,msg):
        if (self.x in locals()) == False:
            self.resolution = msg.info.resolution
            self.originX = msg.info.origin.position.x
            self.originY = msg.info.origin.position.y
            self.width = msg.info.width
            self.height = msg.info.height
            self.data = msg.data
            self.column = int((self.x- msg.info.origin.position.x)/msg.info.resolution)
            self.row = int((self.y- msg.info.origin.position.y)/msg.info.resolution)

        if self.flag == 1:
            self.path = exploration(self.data,self.width,self.height,self.resolution,self.column,self.row,self.originX,self.originY)
            print("[BILGI] HEDEFE GIDILIYOR..")
            self.i = 0
            self.flag = 2

    def timer_callback(self):
        if self.flag == 2:
            twist = Twist()
            v , w = localControl(self.scan)
            if v == None:
                v, w,self.i = pure_pursuit(self.x,self.y,self.yaw,self.path,lookahead_distance,self.i)
            else:
                print("[BILGI] ENGEL TESPİT EDİLDİ]")
            if(abs(self.x - self.path[-1][0]) < target_error and abs(self.y - self.path[-1][1]) < target_error):
                v = 0.0
                w = 0.0
                self.flag = 1
                print("[BILGI] HEDEFE ULASILDI")
                print("[BILGI] YENI HEDEF BELİRLENİYOR")
            twist.linear.x = v
            twist.angular.z = w
            self.publisher.publish(twist)

    def info_callback(self,msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = euler_from_quaternion(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)


def main(args=None):
    rclpy.init(args=args)
    navigation_control = navigationControl()
    rclpy.spin(navigation_control)
    navigation_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
