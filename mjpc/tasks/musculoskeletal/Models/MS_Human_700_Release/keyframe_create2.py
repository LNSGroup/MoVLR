import numpy as np
import matplotlib.pyplot as plt

# with open('./keyframes/sensor_names.txt', 'rb') as file:
#     sensor_names = file.readlines()

folder = 'kick/'

# data = pandas.read_csv(f'./keyframes/{folder}sensor_data_list.csv', header=None).values
qpos_data = np.load(f'./qpos_traj.npy')
# qvel_data = np.load(f'./keyframes/{folder}700qvel.npz')['arr_0']
# print(data.shape)
print('pos')
# print(data[:10, :16])
# fd_vel_1 = (data[1]-data[0]) * 500

vel_gap = 20

# fd_vel = (data[vel_gap]-data[0]) * 1/(0.002*vel_gap)
# fd_vel = np.zeros((len(data)-vel_gap, 16*3))
# i = 0
# while i + vel_gap < len(data):
#     fd_vel[i] = (data[i+vel_gap, :48]-data[i, :48]) * 1/(0.002*vel_gap)
#     i += 1
    

# print(fd_vel[:, 2*3].tolist())


# print(fd_vel_1)
# print(fd_vel_4)
# print(fd_vel_1-fd_vel_4)
# print('vel')
# print(data[:10, 16*3:32*3])

# for k in qpos_data.keys():
#     print(k)
print(qpos_data.shape)
# print(qvel_data.shape)

# for i in range()

# raise NotImplementedError

text_data = []
text_data.append('<mujoco>\n')
text_data.append('\t<keyframe>\n')
for i, d in enumerate(qpos_data):
    # mpos = data[i]
    
    str_data = f'\t\t<key name="walk_{i}"'
    # print(len(qpos))
    # for j in range(48):
    #     str_data += f'{round(mpos[j], 4)} '
    # # for j in range(len(qpos)):
    # #     str_data += f'0.0 '
    # str_data += '"'
    
    qpos = qpos_data[i]
    str_data += f' qpos=" '
    # print(len(qpos))
    # raise NotImplementedError
    for j in range(len(qpos)):
        str_data += f'{round(qpos[j], 4)} '
    # str_data += '"'
    
    # qvel = qvel_data[i]
    # str_data += f' qvel=" '
    # for j in range(len(qvel)):
    #     str_data += f'{round(qvel[j], 4)} '
    str_data += '"/>\n'
    
    
    print(str_data)
    # print('qpos="', end='')
    # for i in range(85):
    #     print("0.0 ", end=')
    # print('" ', end='')
    # print('qvel="', end='')
    # for i in range(85):
    #     print("0.0 ", end='')
    # print('" ', end='')
    # print('/>') 
    text_data.append(str_data)

text_data.append('\t</keyframe>\n')
text_data.append('</mujoco>')

with open(f'./key_qpos_swim.xml', 'w') as file:
    file.writelines(text_data)
    for line in text_data:
        file.write(line + '\n')
# raise NotImplementedError

