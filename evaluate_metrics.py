# -*- coding: utf-8 -*-
"""
Evaluates a trained call_id.py/SincNet checkpoint on its test set and
writes accuracy / roc_auc / precision / recall / f1 / FPR / FNR /
top3 / top5 / roc_auc_mean_exp / trainable_params to <output_folder>/metrics.res, matching
the metrics reported in Table 1 of Bravo Sanchez et al. 2021.

How to run it:
python evaluate_metrics.py --cfg=mod_cfg/mod_nips4bplus_bird_species.cfg
"""

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from metrics_utils import compute_metrics

from data_io import read_conf, str_to_bool
from dnn_models import MLP
from dnn_models import SincNet as CNN

options = read_conf()

tr_lst = options.tr_lst
te_lst = options.te_lst
data_folder = options.data_folder + '/'
output_folder = options.output_folder

fs = int(options.fs)
cw_len = int(options.cw_len)
cw_shift = int(options.cw_shift)

cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act = list(map(str, options.cnn_act.split(',')))
cnn_drop = list(map(float, options.cnn_drop.split(',')))

fc_lay = list(map(int, options.fc_lay.split(',')))
fc_drop = list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp = str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp = str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm = list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm = list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act = list(map(str, options.fc_act.split(',')))

class_lay = list(map(int, options.class_lay.split(',')))
class_drop = list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp = str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp = str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm = list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm = list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act = list(map(str, options.class_act.split(',')))

wlen = int(fs*cw_len/1000.00)
wshift = int(fs*cw_shift/1000.00)
# Eval-only batch size; frames are scored independently in eval mode, so this
# only changes speed (original was 128)
Batch_dev = 1024

wav_lst_te = pd.read_csv(te_lst)
snt_te = len(wav_lst_te)
n_classes = class_lay[-1]

CNN_arch = {'input_dim': wlen, 'fs': fs, 'cnn_N_filt': cnn_N_filt,
            'cnn_len_filt': cnn_len_filt, 'cnn_max_pool_len': cnn_max_pool_len,
            'cnn_use_laynorm_inp': cnn_use_laynorm_inp, 'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
            'cnn_use_laynorm': cnn_use_laynorm, 'cnn_use_batchnorm': cnn_use_batchnorm,
            'cnn_act': cnn_act, 'cnn_drop': cnn_drop}
CNN_net = CNN(CNN_arch)
CNN_net.cuda()

DNN1_arch = {'input_dim': CNN_net.out_dim, 'fc_lay': fc_lay, 'fc_drop': fc_drop,
             'fc_use_batchnorm': fc_use_batchnorm, 'fc_use_laynorm': fc_use_laynorm,
             'fc_use_laynorm_inp': fc_use_laynorm_inp, 'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
             'fc_act': fc_act}
DNN1_net = MLP(DNN1_arch)
DNN1_net.cuda()

DNN2_arch = {'input_dim': fc_lay[-1], 'fc_lay': class_lay, 'fc_drop': class_drop,
             'fc_use_batchnorm': class_use_batchnorm, 'fc_use_laynorm': class_use_laynorm,
             'fc_use_laynorm_inp': class_use_laynorm_inp, 'fc_use_batchnorm_inp': class_use_batchnorm_inp,
             'fc_act': class_act}
DNN2_net = MLP(DNN2_arch)
DNN2_net.cuda()

checkpoint_load = torch.load(output_folder+'/model_raw.pkl', map_location='cuda')
CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])

trainable_params = (sum(p.numel() for p in CNN_net.parameters() if p.requires_grad)
                     + sum(p.numel() for p in DNN1_net.parameters() if p.requires_grad)
                     + sum(p.numel() for p in DNN2_net.parameters() if p.requires_grad))

CNN_net.eval()
DNN1_net.eval()
DNN2_net.eval()

y_true = np.zeros(snt_te, dtype=int)
y_pred = np.zeros(snt_te, dtype=int)
y_score = np.zeros((snt_te, n_classes))
y_score_exp = np.zeros((snt_te, n_classes))

# Whole-file scores, computed once per unique file (the test list has
# several tag rows per file and each row scores the same whole file)
file_scores = {}
file_scores_exp = {}

with torch.no_grad():
    for i in range(snt_te):
        fname = wav_lst_te.loc[i, 'file']
        lab_batch = wav_lst_te.loc[i, 'label']

        if fname not in file_scores:
            [signal, fs_i] = sf.read(data_folder+fname)
            signal = torch.from_numpy(signal).float().cuda().contiguous()

            N_fr = int((signal.shape[0]-wlen)/wshift)

            # Same frames as the original while-loop (start k*wshift while
            # start+wlen < len, strictly), built in one unfold call. As before,
            # pout has N_fr+1 rows and any row not covered by a frame stays 0.
            n_frames = max(0, -(-(signal.shape[0]-wlen)//wshift))
            frames = signal.unfold(0, wlen, wshift)[:n_frames]
            pout = torch.zeros(N_fr+1, n_classes).float().cuda().contiguous()
            for beg in range(0, n_frames, Batch_dev):
                inp = frames[beg:beg+Batch_dev].contiguous()
                pout[beg:beg+inp.shape[0], :] = DNN2_net(DNN1_net(CNN_net(inp)))

            # Sentence-level decision matches call_id.py's own validation loop
            # exactly: argmax(sum(pout, dim=0)), i.e. summed log-softmax scores
            # across all frames of the whole file (not an average of per-frame
            # probabilities, which -- since the tagged call can be a small
            # fraction of a several-second recording -- gets diluted by
            # background/silence frames and collapses accuracy toward random).
            # softmax() of that same sum is monotonic with it (same argmax) and
            # gives a valid probability vector for ROC AUC / top-k.
            file_scores[fname] = torch.sum(pout, dim=0)
            # "Mean Exp" (Table S7): mean over frames of exp(LogSoftmax output)
            file_scores_exp[fname] = torch.exp(pout[:n_frames]).mean(dim=0)

        sent_scores = file_scores[fname]
        sent_probs = torch.softmax(sent_scores, dim=0)

        y_true[i] = int(lab_batch)
        y_pred[i] = int(torch.argmax(sent_probs).item())
        y_score[i, :] = sent_probs.cpu().numpy()
        y_score_exp[i, :] = file_scores_exp[fname].cpu().numpy()

m = compute_metrics(y_true, y_pred, y_score, y_score_exp, n_classes)
m['trainable_params'] = trainable_params

with open(output_folder+"/metrics.res", "w") as f:
    for k, v in m.items():
        f.write(("%s=%d\n" if k == 'trainable_params' else "%s=%.4f\n") % (k, v))

# saved for later plots (confusion matrix, ROC curves)
np.savez(output_folder+"/predictions.npz", y_true=y_true, y_pred=y_pred,
         y_score=y_score, y_score_exp=y_score_exp)

print(" ".join(("%s=%d" if k == 'trainable_params' else "%s=%.4f") % (k, v) for k, v in m.items()))
