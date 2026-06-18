filename = 'Body_Torso_Simple.xml'

with open(filename, 'r') as file:
    data = file.readlines()

data_new = []
for line in data:
    # print(line)
    segs = line.split(' ')
    # print(segs)
    # break
    for i, seg in enumerate(segs):
        if 'name' in seg:
            print('old one', seg)
            segs[i] = seg[:-1] + '_2"'
            print('new one', segs[i])   
    # print(line)
    # print(segs)
    new_line = ' '.join(segs)
    print(new_line)
    data_new.append(new_line)

with open(f'{filename[:-4]}_2.xml', 'w') as file:
    file.writelines(data_new)
            
# print(data)