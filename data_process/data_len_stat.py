from include import *
from helper_code import *

image_file = glob.glob(DATA_ROOT_PATH + '/CinC2021/all_data/*.mat')

len_0 = 0
len_20 = 0
len_30 = 0
len_60 = 0
len_180 = 0
len_300 = 0
len_600 = 0
len_max = 0

for i in range(len(image_file)):
    data = sio.loadmat(image_file[i])
    header = load_header(image_file[i].replace('mat', 'hea'))
    sampr = get_frequency(header)
    data = data['val']
    data_len = len(data[0]) / sampr
    if data_len < 20 :
        len_20 += 1
    elif data_len < 30 :
        len_30 += 1
    elif data_len < 60 :
        len_60 += 1
    elif data_len < 180 :
        len_180 += 1
    elif data_len < 300 :
        len_300 += 1
    elif data_len < 600 :
        len_600 += 1
    else:
        len_max += 1

print('len_20:', len_20)
print('len_30:', len_30)
print('len_60:', len_60)
print('len_180:', len_180)
print('len_300:', len_300)
print('len_600:', len_600)
print('len_max:', len_max)