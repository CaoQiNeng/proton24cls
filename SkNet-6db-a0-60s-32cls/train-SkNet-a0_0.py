import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common  import *
from model.model_AttentionInCNN import *
from dataset_27cls_60s import *

def metric(truth, predict):
    truth_for_cls = np.sum(truth, axis=0) + 1e-11
    predict_for_cls = np.sum(predict, axis=0) + 1e-11

    # TP
    count = truth + predict
    count[count != 2] = 0
    TP = np.sum(count, axis=0) / 2

    precision = TP / predict_for_cls
    recall = TP / truth_for_cls

    return precision, recall

#------------------------------------
def do_valid(net, valid_loader):
    valid_loss = 0
    valid_predict = []
    valid_truth = []
    infors = []
    valid_num = 0

    for t, (input, truth, infor) in enumerate(valid_loader):
        batch_size = len(infor)

        infors.append(infor)

        net.eval()
        input  = input.cuda()
        truth  = truth.cuda()

        with torch.no_grad():
            logit = data_parallel(net, input) #net(input)
            probability = torch.sigmoid(logit)

            loss = F.binary_cross_entropy(probability, truth)

        valid_predict.append(probability.cpu().numpy())
        valid_truth.append(truth.cpu().numpy().astype(int))

        #---
        valid_loss += loss.cpu().numpy() * batch_size

        valid_num  += batch_size

        print('\r %8d / %d'%(valid_num, len(valid_loader.dataset)),end='',flush=True)

        pass  #-- end of one data loader --

    # assert(valid_num == len(valid_loader.dataset))
    valid_loss = valid_loss / (valid_num+1e-8)

    infors = np.hstack(infors)
    valid_truth = np.vstack(valid_truth)
    valid_predict = np.vstack(valid_predict)
    valid_predict_class = valid_predict>0.5

    valid_predict_class[:, -3] = 0

    valid_precision, valid_recall = metric(valid_truth, valid_predict_class.astype(int))

    return valid_loss, valid_precision, valid_recall

def run_train():
    fold = 0
    global out_dir
    out_dir = ROOT_PATH + '/CinC2021_logs/result-SkNet--6db-a%d-60s-27cls'%(fold)
    initial_checkpoint = None

    schduler = NullScheduler(lr=0.1)
    iter_accum = 1
    batch_size = 18 #8

    ## setup  -----------------------------------------------------------------------------
    for f in ['checkpoint','train','valid','backup'] : os.makedirs(out_dir +'/'+f, exist_ok=True)

    # backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = CinCDataset(
        mode='train',
        csv='train.csv',
        split='train_a%d_38791.npy' % (fold),
        fold=fold
    )

    train_loader = DataLoader(
        train_dataset,
        sampler     = RandomSampler(train_dataset),
        # sampler = CustomSampler(train_dataset),
        # shuffle=True,
        batch_size=batch_size,
        drop_last=False,
        num_workers=20,
        pin_memory=True,
        collate_fn=null_collate
    )

    val_dataset = CinCDataset(
        mode='train',
        csv='train.csv',
        split='valid_a%d_4310.npy' % (0),
        fold=0
    )

    valid_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=10,
        pin_memory=True,
        collate_fn=null_collate
    )

    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(val_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = mySKNetBlock(12, out_channels=len(class_map)).cuda()
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)

    if initial_checkpoint is not None:
        state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)

        # net.load_state_dict(state_dict,strict=False)
        net.load_state_dict(state_dict,strict=True)  #True

    log.write('net=%s\n'%(type(net)))
    log.write('\n')

    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    #optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.0, weight_decay=0.0)

    num_iters   = 3000*1000
    iter_smooth = 400
    iter_log    = 400
    iter_valid  = 400
    iter_save   = [0, num_iters-1]\
                   + list(range(0, num_iters, 400))#1*1000

    start_iter = 0
    start_epoch= 0
    rate       = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace('_model.pth','_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint  = torch.load(initial_optimizer)
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']
            #optimizer.load_state_dict(checkpoint['optimizer'])
        pass

    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('schduler\n  %s\n'%(schduler))
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('   batch_size=%d,  iter_accum=%d\n'%(batch_size,iter_accum))
    log.write('   experiment  = %s\n' % str(__file__.split('/')[-2:]))
    log.write('----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
    log.write('mode    rate    iter  epoch | loss  | 270492004 | 164889003 | 164890007 | 426627000 | 713427006 | 713426002 | 445118002 | 39732003  | 164909002 | 251146004 | 698252002 | 10370003  | 284470004 | 427172004 | 164947007 | 111975006 | 164917005 | 47665007  | 59118001  | 427393009 | 426177001 | 426783006 | 427084000 | 63593006  | 164934002 | 59931005  | 17338001  | time        \n')
    log.write('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
              # train  0.01000   0.5   0.2 | 1.11  | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 0 hr 05 min
    def message(rate, iter, epoch, loss, precision, recall, mode='print', train_mode = 'train'):
        precision_recall = []
        for p, r in zip(precision, recall) :
            precision_recall.append(p)
            precision_recall.append(r)

        if mode==('print'):
            asterisk = ' '
        if mode==('log'):
            asterisk = '*' if iter in iter_save else ' '

        text = \
            '%s   %0.5f %5.1f%s %4.1f | '%(train_mode, rate, iter/1000, asterisk, epoch,) +\
            '%4.3f | '%loss +\
            '%0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | '%(*precision_recall, ) +\
            '%s' % (time_to_str((timer() - start_timer),'min'))

        return text

    #----
    train_loss = 0
    train_precision = [0 for i in range(len(class_map))]
    train_recall = [0 for i in range(len(class_map))]
    iter = 0
    i    = 0

    start_timer = timer()
    while  iter<num_iters:
        train_predict_list = []
        train_truth_list = []
        sum_train_loss = 0
        sum_train = 0

        optimizer.zero_grad()
        for t, (input, truth, infor) in enumerate(train_loader):
            batch_size = len(infor)
            iter  = i + start_iter
            epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch

            #if 0:
            if (iter % iter_valid==0):
                valid_loss, valid_precision, valid_recall = do_valid(net, valid_loader) #
                pass

            if (iter % iter_log==0):
                print('\r',end='',flush=True)
                print(message(rate, iter, epoch, train_loss, train_precision, train_recall, mode='log', train_mode='train'))
                log.write(message(rate, iter, epoch, valid_loss, valid_precision, valid_recall,mode='log', train_mode='valid'))
                log.write('\n')

            top_loss = np.array([0.15 for i in range(10)])
            top_F2 = [0.83 for i in range(10)]
            if iter in iter_save:
                current_loss = valid_loss

                top_loss = np.sort(top_loss)

                if current_loss < top_loss[9]:
                    top_loss[9] = current_loss

                torch.save({
                    #'optimizer': optimizer.state_dict(),
                    'iter'     : iter,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))
                if iter!=start_iter:
                    torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(iter))
                    pass

            # learning rate schduler -------------
            lr = schduler(iter)
            if lr<0 : break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            #net.set_mode('train',is_freeze_bn=True)
            net.train()
            input = input.cuda()
            truth = truth.cuda()

            logit = data_parallel(net, input)
            probability = torch.sigmoid(logit)

            loss = F.binary_cross_entropy(probability, truth)

            loss.backward()
            loss = loss.detach().cpu().numpy()
            if (iter % iter_accum)==0:
                optimizer.step()
                optimizer.zero_grad()

            predict = probability.cpu().detach().numpy()
            truth = truth.cpu().numpy().astype(int)
            batch_precision, batch_recall= metric(truth, (predict>0.5).astype(int))

            # print statistics  --------
            batch_loss      = loss
            train_predict_list.append(predict)
            train_truth_list.append(truth)
            sum_train_loss += loss * batch_size
            sum_train      += batch_size
            if iter%iter_smooth == 0:
                train_loss = sum_train_loss / (sum_train+1e-12)
                train_predict_list = np.vstack(train_predict_list)
                train_truth_list = np.vstack(train_truth_list)
                train_precision, train_recall = metric(train_truth_list, (train_predict_list>0.5).astype(int))

                train_predict_list = []
                train_truth_list = []
                sum_train_loss = 0
                sum_train      = 0

            # print(batch_loss)
            print('\r',end='',flush=True)
            print(message(rate, iter, epoch, batch_loss, batch_precision, batch_recall, mode='log', train_mode='train'), end='',flush=True)
            i=i+1

        pass  #-- end of one data loader --
    pass #-- end of all iterations --
    log.write('\n')

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()
