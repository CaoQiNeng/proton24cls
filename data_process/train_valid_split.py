from common import *

def run_make_train_5F():
    image_file = glob.glob(DATA_ROOT_PATH + '/proton/overall/*.json')
    image_file = [(os.path.split(i)[-1]).split('.')[0] for i in image_file]

    random.shuffle(image_file)


    # 12568
    num_fold = 5
    num_valid = int(len(image_file) * 0.1)

    for n in range(num_fold):
        valid = image_file[n * num_valid:(n + 1) * num_valid]
        train = list(set(image_file) - set(valid))

        print(set(train).intersection(valid))
        np.save(DATA_ROOT_PATH + '/proton/split/train_a%d_%d.npy' % (n, len(train)), train)
        np.save(DATA_ROOT_PATH + '/proton/split/valid_a%d_%d.npy' % (n, len(valid)), valid)

run_make_train_5F()