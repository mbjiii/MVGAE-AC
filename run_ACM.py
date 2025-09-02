import random
import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np


from utils.pytorchtools import EarlyStopping
from utils.data import load_ACM_data
from utils.tools import  evaluate_results_nc2, index_generator, parse_minibatch
from utils.utils import row_normalize
from model.MAGNN_nc_mb import MAGNN_nc_mb

# Params
out_dim = 3
dropout_rate = 0.7
lr = 0.003
weight_decay = 0.001
etypes_lists = [[0, 1], [2, 3]]
num_metapaths = 2
num_edge_type = 4

seed = 789
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def run_model_ACM( dataset,num_layers,hidden_dim, num_heads, attn_vec_dim, rnn_type,
                   num_epochs, patience, repeat, save_postfix,args):

    adjlists, edge_metapath_indices_lists, adjM, type_mask, labels, train_val_test_idx, A, X,adj_full = load_ACM_data()





    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_nodes = A[0].shape[0]
    num_rel = len(A)
    A = row_normalize(A)
    in_dim_list = [X[i].shape[1] for i in range(len(X))]
    RGCN_argument = [args.trans_dim, args.hidden, args.emb_dim, args.emb_alpha, args.bases, args.drop, args.using_cuda]
    X = [X[i].to(device) for i in range(len(X))]
    labels = torch.LongTensor(labels).to(device)

    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    feats_opt = args.feats_opt
    feats_opt = list(feats_opt)
    feats_opt = list(map(int, feats_opt))



    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    nmi_mean_list = []
    nmi_std_list = []
    ari_mean_list = []
    ari_std_list = []



    for cur_repeat in range(repeat):
        print('cur_repeat = {}  start ==============================================================='.format(cur_repeat))

        net = MAGNN_nc_mb(dataset,num_metapaths, num_edge_type, etypes_lists, in_dim_list, hidden_dim, out_dim, num_heads,
                          attn_vec_dim,num_rel,adj_full, A, X,RGCN_argument,feats_opt,args.gamma,  args.layers,
                          args.emb_alpha, args.k_KNN, args.ratio_mask, rnn_type, dropout_rate)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        target_node_indices = np.where(type_mask == 0)[0],
        print('model init finish\n')

        # training loop
        print('training...')
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True,
                                       save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
        train_idx_generator = index_generator(batch_size=args.batch_size, indices=train_idx)
        val_idx_generator = index_generator(batch_size=args.batch_size, indices=val_idx, shuffle=False)
        dur1 = []
        dur2 = []
        dur3 = []
        T = 0
        for epoch in range(num_epochs):
            t0 = time.time()

            # training forward
            net.train()
            train_loss_avg = 0
            for iteration in range(train_idx_generator.num_iterations()):
                train_idx_batch = train_idx_generator.next()
                train_idx_batch.sort()
                train_g_list, train_indices_list, train_idx_batch_mapped_list = parse_minibatch(
                    adjlists, edge_metapath_indices_lists, train_idx_batch, device, args.neighbor_samples)
                for i in range(len(train_g_list)):
                    train_g_list[i] = train_g_list[i].to(device)

                logits, embeddings,emb_loss,recon_loss,_ = net((train_g_list, type_mask, train_indices_list, train_idx_batch_mapped_list))
                logp = F.log_softmax(logits, 1)
                train_loss = F.nll_loss(logp, labels[train_idx_batch]) + args.alpha*emb_loss + args.beta*recon_loss
                train_loss_avg +=train_loss.item()

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            train_loss_avg /= train_idx_generator.num_iterations()
            t1 = time.time()
            dur1.append(t1 - t0)




            t2 = time.time()
            dur2.append(t2 - t1)
            T+=(t1-t0)+(t2-t1)
            # validation forward
            net.eval()
            val_loss_avg = 0
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    val_idx_batch = val_idx_generator.next()
                    val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                        adjlists, edge_metapath_indices_lists, val_idx_batch, device, args.neighbor_samples)
                    for i in range(len(val_g_list)):
                        val_g_list[i] = val_g_list[i].to(device)

                    logits, embeddings,emb_loss,recon_loss,_ = net(
                                    (val_g_list, type_mask, val_indices_list, val_idx_batch_mapped_list))
                    logp = F.log_softmax(logits, 1)
                    val_loss = F.nll_loss(logp, labels[val_idx_batch])+args.alpha*emb_loss+args.beta*recon_loss
                    val_loss_avg += val_loss.item()
                val_loss_avg /= val_idx_generator.num_iterations()


            t3 = time.time()
            dur3.append(t3 - t2)


            print(
                "Epoch {:05d} | Train_Loss {:.4f} | Val_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f} | CUDA(GB) {:.4f}".format(
                    epoch, train_loss_avg, val_loss_avg, np.mean(dur1), np.mean(dur2), np.mean(dur3), torch.cuda.max_memory_allocated()/1024**3))

            # early stopping
            early_stopping(val_loss_avg, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f" Max GPU memory: {max_mem:.2f} GB")

        print('\ntesting...')
        test_idx_generator = index_generator(batch_size=args.batch_size, indices=test_idx, shuffle=False)
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
        net.eval()
        test_embeddings = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                test_idx_batch = test_idx_generator.next()
                test_g_list, test_indices_list, test_idx_batch_mapped_list = parse_minibatch(
                    adjlists, edge_metapath_indices_lists, test_idx_batch, device,args.neighbor_samples)
                for i in range(len(test_g_list)):
                    test_g_list[i] = test_g_list[i].to(device)

                logits, embeddings,_,_,_ = net((test_g_list, type_mask, test_indices_list, test_idx_batch_mapped_list))
                test_embeddings.append(embeddings)

            test_embeddings = torch.cat(test_embeddings, 0)
            embeddings = test_embeddings.detach().cpu().numpy()
            svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std,  = evaluate_results_nc2(
                embeddings, labels[test_idx].cpu().numpy(), num_classes=out_dim)





        svm_macro_f1_lists.append(svm_macro_f1_list)
        svm_micro_f1_lists.append(svm_micro_f1_list)
        nmi_mean_list.append(nmi_mean)
        nmi_std_list.append(nmi_std)
        ari_mean_list.append(ari_mean)
        ari_std_list.append(ari_std)


    svm_macro_f1_lists = np.transpose(np.array(svm_macro_f1_lists), (1, 0, 2))
    svm_micro_f1_lists = np.transpose(np.array(svm_micro_f1_lists), (1, 0, 2))
    nmi_mean_list = np.array(nmi_mean_list)
    nmi_std_list = np.array(nmi_std_list)
    ari_mean_list = np.array(ari_mean_list)
    ari_std_list = np.array(ari_std_list)





if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the IMDB dataset')
    ap.add_argument('--dataset',type=str,default = 'ACM')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=10, help='Patience. Default is 10.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument("--hidden", type=int, default=64, help="Number of hidden units.")
    ap.add_argument("--emb_dim", type=int, default=64, help="dim of the embedding")
    ap.add_argument("--trans_dim", type=int, default=64, help="dim of the embedding")



    ap.add_argument('--save-postfix', default='ACM', help='Postfix for the saved model and result. Default is IMDB.')
    ap.add_argument('--layers', type=int, default=2, help='Number of layers. Default is 2.')
    ap.add_argument("--drop", type=float, default=0.3, help="Dropout of RGCN")
    ap.add_argument("--bases", type=int, default=0, help="R-GCN bases")
    ap.add_argument("--emb_alpha", type=float, default=0.5, help="ratio between R-GAT emb and structure-guide emb")
    ap.add_argument("--gamma", type=float, default=3.0, help="hyper-parameter: scaling factor")
    ap.add_argument("--k-KNN", type=int, default=5, help="number of  KNN")
    ap.add_argument("--ratio-mask", type=float, default=0.7, help="ratio of masked edges")
    ap.add_argument("--alpha", type=float, default=2.0, help="ratio of emb_loss")
    ap.add_argument("--beta", type=float, default=0.9, help="ratio of recon_loss")
    ap.add_argument('--batch-size', type=int, default=512, help='Batch size. Default is 8.')
    ap.add_argument('--neighbor-samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')

    ap.add_argument(
        "--no_cuda", action="store_true", default=False, help="Enables CUDA training."
    )
    ap.add_argument("--feats_opt", type=str, default='011', help='0100 means 1 type nodes use our processed feature')
    args = ap.parse_args()
    args.using_cuda = not args.no_cuda and torch.cuda.is_available()

    run_model_ACM( args.dataset,args.layers, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type,
                   args.epoch, args.patience, args.repeat, args.save_postfix,args)
