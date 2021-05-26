from common import *
from scipy import signal
from helper_code import *

#--------------
DATA_DIR = DATA_ROOT_PATH + '/proton'
# class_map = pd.read_csv(DATA_DIR + '/evaluation-2021/dx_mapping_scored.csv')['SNOMED CT Code'].to_numpy()
class_map = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])

class CinCDataset(Dataset):
    def __init__(self, split, mode, csv, fold):
        self.split   = split
        self.csv     = csv
        self.mode    = mode

        self.Recording = np.load(DATA_DIR + '/split/%s' % split)

        self.num_image = len(self.Recording)

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\n'
        string += '\tmode     = %s\n'%self.mode
        string += '\tsplit    = %s\n'%self.split
        string += '\tcsv      = %s\n'%str(self.csv)
        string += '\tnum_image = %d\n'%self.num_image
        return string

    def __len__(self):
        return self.num_image

    def __getitem__(self, index):
        ecg_id = self.Recording[index]
        # print(ecg_id)
        if ecg_id == 'zhuyishengdat_12_1':
            a = 1

        header = load_header(DATA_DIR + '/overall/%s.hea' % ecg_id)
        #
        sampr = int(get_sampr(header))
        # sampr = 256

        with open(DATA_DIR + '/overall/%s.json' % ecg_id, 'r') as load_f:
            old_temp_ecg = json.load(load_f)['ecgyr']
        # old_temp_ecg = np.array(old_temp_ecg / 1000)
        temp_ecg = []
        # for i in range(len(old_temp_ecg)):
        temp_ecg = resample(old_temp_ecg[:], sampr)

        # temp_ecg = np.array(temp_ecg)

        ecg = np.zeros((1,9000), dtype=np.float32)
        ecg[0][-temp_ecg.shape[0]:] = temp_ecg[-9000:]

        infor = Struct(
            index  = index,
            ecg_id = ecg_id,
        )

        labels = get_labels(header)

        label = np.zeros(len(class_map))

        for l in labels:
            l_index = np.where(class_map == int(l))[0]
            if len(l_index) > 0:
                label[l_index] = 1

        return ecg, label, infor

class CustomSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = self.dataset.First_label

        self.AF_index  = np.where(label == 0)[0]
        self.I_AVB_index = np.where(label == 1)[0]
        self.LBBB_index = np.where(label == 2)[0]
        self.Normal_index = np.where(label == 3)[0]
        self.PAC_index = np.where(label == 4)[0]
        self.PVC_index = np.where(label == 5)[0]
        self.RBBB_index = np.where(label == 6)[0]
        self.STD_index = np.where(label == 7)[0]
        self.STE_index = np.where(label == 8)[0]

        #assume we know neg is majority class
        num_RBBB = len(self.RBBB_index)
        self.length = 9 * num_RBBB

    def __iter__(self):
        RBBB = self.RBBB_index.copy()
        np.random.shuffle(RBBB)
        # num_RBBB = len(self.RBBB_index)

        AF = np.random.choice(self.AF_index, len(self.AF_index), replace=False)
        I_AVB = np.random.choice(self.I_AVB_index, len(self.I_AVB_index), replace=False)
        LBBB = np.random.choice(self.LBBB_index, len(self.PAC_index), replace=True)
        Normal = np.random.choice(self.Normal_index, len(self.Normal_index), replace=False)
        PAC = np.random.choice(self.PAC_index, len(self.PAC_index), replace=True)
        PVC = np.random.choice(self.PVC_index, len(self.PVC_index), replace=False)
        STD = np.random.choice(self.STD_index, len(self.PAC_index), replace=False)
        STE = np.random.choice(self.STE_index, len(self.STD_index), replace=True)

        l = np.hstack([AF,I_AVB,LBBB,Normal,PAC,PVC,RBBB,STD,STE])
        np.random.shuffle(l)

        return iter(l)

    def __len__(self):
        return self.length

class BalanceClassSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = self.dataset.First_label

        self.AF_index  = np.where(label == 0)[0]
        self.I_AVB_index = np.where(label == 1)[0]
        self.LBBB_index = np.where(label == 2)[0]
        self.Normal_index = np.where(label == 3)[0]
        self.PAC_index = np.where(label == 4)[0]
        self.PVC_index = np.where(label == 5)[0]
        self.RBBB_index = np.where(label == 6)[0]
        self.STD_index = np.where(label == 7)[0]
        self.STE_index = np.where(label == 8)[0]

        #assume we know neg is majority class
        num_RBBB = len(self.RBBB_index)
        self.length = 9 * num_RBBB

    def __iter__(self):
        RBBB = self.RBBB_index.copy()
        np.random.shuffle(RBBB)
        num_RBBB = len(self.RBBB_index)

        AF = np.random.choice(self.AF_index, num_RBBB, replace=True)
        I_AVB = np.random.choice(self.I_AVB_index, num_RBBB, replace=True)
        LBBB = np.random.choice(self.LBBB_index, num_RBBB, replace=True)
        Normal = np.random.choice(self.Normal_index, num_RBBB, replace=True)
        PAC = np.random.choice(self.PAC_index, num_RBBB, replace=True)
        PVC = np.random.choice(self.PVC_index, num_RBBB, replace=True)
        STD = np.random.choice(self.STD_index, num_RBBB, replace=True)
        STE = np.random.choice(self.STE_index, num_RBBB, replace=True)

        l = np.stack([AF,I_AVB,LBBB,Normal,PAC,PVC,RBBB,STD,STE]).T
        l = l.reshape(-1)

        return iter(l)

    def __len__(self):
        return self.length

# Find Challenge files.
def find_challenge_files(label_directory, output_directory):
    label_files = list()
    output_files = list()
    for f in sorted(os.listdir(label_directory)):
        F = os.path.join(label_directory, f) # Full path for label file
        if os.path.isfile(F) and F.lower().endswith('.hea') and not f.lower().startswith('.'):
            root, ext = os.path.splitext(f)
            g = root + '.csv'
            G = os.path.join(output_directory, g) # Full path for corresponding output file
            if os.path.isfile(G):
                label_files.append(F)
                output_files.append(G)
            else:
                raise IOError('Output file {} not found for label file {}.'.format(g, f))

    if label_files and output_files:
        return label_files, output_files
    else:
        raise IOError('No label or output files found.')

def resample(data, sampr, after_Hz=300):
    data_len = len(data)
    propessed_data = signal.resample(data, int(data_len * (after_Hz / sampr)))

    return propessed_data

def null_collate(batch):
    batch_size = len(batch)

    input = []
    label = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        label.append(batch[b][1])
        infor.append(batch[b][-1])

    label = torch.from_numpy(np.stack(label)).float()
    input = torch.from_numpy(np.stack(input)).float()

    return input, label, infor

def run_check_DataSet():
    fold = 0
    val_dataset = CinCDataset(
        mode='train',
        csv='train.csv',
        split='train_a%d_10346.npy' % (0),
        fold=0
    )

    # train_dataset = CinCDataset(
    #     mode='train',
    #     csv='train.csv',
    #     split='train_a%d_38791.npy' % (fold),
    #     fold=fold
    # )


    a = 0
    for t, (input, truth, infor) in enumerate(val_dataset):
        a += 1

        if np.sum(truth) > 0 :
            # print(infor.ecg_id)
            print(truth)



# main #################################################################
if __name__ == '__main__':
    run_check_DataSet()

    print('\nsucess!')