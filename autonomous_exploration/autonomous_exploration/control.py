import rclpy
from rclpy.node import Node
import numpy as np
import heapq
from nav_msgs.msg import OccupancyGrid , Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math
import scipy.interpolate as si
import yaml

with open("src/autonomous_exploration/config/params.yaml", 'r') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

lookahead_distance = params["lookahead_distance"]
speed = params["speed"]
expansion_size = params["expansion_size"]
group_length = params["group_length"]
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


    return False

def bspline_planning(x, y, sn):
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

    return rx, ry

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
    for group in list(groups):
        if len(groups[group]) <= group_length:
            del groups[group]
    return groups

def findClosestGroup(matrix,groups, current,resolution,originX,originY):
    min = 10000000
    target = None
    for value in groups.items():
        middle = value[1][int(len(value[1])/2)]
        path = astar(matrix, current, middle)
        if path == False:
            continue
        path = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
        for i in range(len(path)):
            path[i] = (path[i][0],path[i][1])
            points = np.array(path)
        differences = np.diff(points, axis=0)
        distances = np.hypot(differences[:,0], differences[:,1])
        total_distance = np.sum(distances)
        if total_distance < min and total_distance > target_error*2:
            target = middle
            min = total_distance
        if target == None:
            target = value[1][-1]
    return target[0],target[1]

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
        print("[BILGI] KESIF MODU AKTIF")
        print("[BILGI] YENI HEDEF BELIRLENIYOR")

    def scan_callback(self,msg):
        self.scan = msg.ranges

    def listener_callback(self,msg):
        if self.flag == 1:
            self.resolution = msg.info.resolution
            self.originX = msg.info.origin.position.x
            self.originY = msg.info.origin.position.y
            if self.x == None:
                return
            column = int((self.x- msg.info.origin.position.x)/msg.info.resolution)
            row = int((self.y- msg.info.origin.position.y)/msg.info.resolution)
            width = msg.info.width
            height = msg.info.height
            data = costmap(msg.data,width,height,msg.info.resolution)
            data[row][column] = 0 #Robot Anlık Konum
            data[data > 5] = 1 # 0 olanlar gidilebilir yer, 100 olanlar kesin engel
            data = frontierB(data)
            data,groups = assign_groups(data)
            groups = deleteGroups(groups)
            if len(groups) == 0:
                print("[BILGI] KESIF TAMAMLANDI")
                self.flag = 3
                return
            data[data < 0] = 1 #-0.05 olanlar bilinmeyen yer
            #data icerigi 0: gidilebilir, 1: engel
            rowH,columnH = findClosestGroup(data,groups,(row,column),self.resolution,self.originX,self.originY)
            path = astar(data,(row,column),(rowH,columnH))
            path = path + [(row,column)]
            path = path[::-1]
            pathB = path
            pathB = [(p[1]*self.resolution+self.originX,p[0]*self.resolution+self.originY) for p in pathB]
            pathB = np.array(pathB)
            pathX = pathB[:,0]
            pathY = pathB[:,1]
            pathX,pathY = bspline_planning(pathX,pathY,len(pathX)*5)
            self.path = [(pathX[i],pathY[i]) for i in range(len(pathX))]
            print("[BILGI] HEDEFE GIDILIYOR..")
            self.i = 0
            self.flag = 2

    def timer_callback(self):
        if self.flag == 2:
            twist = Twist()
            for i in range(60):
                if self.scan[i] < robot_r:
                    twist.linear.x = 0.2
                    twist.angular.z = -math.pi/4
                    self.publisher.publish(twist)
                    print("[BILGI] LOKAL ENGEL TESPİT EDİLDİ")
                    return
            for i in range(300,360):
                if self.scan[i] < robot_r:
                    twist.linear.x = 0.2
                    twist.angular.z = math.pi/4
                    self.publisher.publish(twist)
                    print("[BILGI] LOKAL ENGEL TESPİT EDİLDİ")
                    return
            twist.linear.x , twist.angular.z,self.i = pure_pursuit(self.x,self.y,self.yaw,self.path,lookahead_distance,self.i)
            if(abs(self.x - self.path[-1][0]) < target_error and abs(self.y - self.path[-1][1]) < target_error):
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.flag = 1
                print("[BILGI] HEDEFE ULASILDI")
                print("[BILGI] YENI HEDEF BELIRLENIYOR")
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

