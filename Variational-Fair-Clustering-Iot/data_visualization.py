import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib import pyplot as plt
import os.path as osp
import pandas as pd


def plot_clusters_vs_lambda(X_org, demograph, l,filename,dataset, lmbda, fairness_error, cluster_option):


        working_dir = osp.dirname(osp.abspath(__file__))
        path = osp.join(working_dir, 'data/Sensor/cluster_output/')
        print(" plot_clusters_vs_lambda")

        all_data = []

        if(lmbda%500 == 0): #added only for Kmeans as it has many iterations
            K = max(l) +1
            COLORSX = np.array(['rD', 'mD', 'gP', 'bP', 'yP', 'gD','rP', 'mP', 'yD', 'rX','gX', 'bX', 'yX', 'mX', 'rD', 'mD', 'gP', 'bP', 'yP', 'yD'])
                                #'gW','rW', 'mW', 'yW', 'bR', 'yR', 'gF','rG', 'mG', 'yG'])
            plt.figure(1,figsize=(6.4,4.8))
            plt.ion()
            plt.clf()

            group = ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 5', 'cluster 6', 'cluster 7', 'cluster 8', 'cluster 9', 'cluster 10',
                     'cluster 11', 'cluster 12',
                     'cluster 13', 'cluster 14','cluster 15','cluster 16', 'cluster 17','cluster 18','cluster 19', 'cluster 20']

            for k in range(K):
                idx = np.asarray(np.where(l == k)).squeeze()
                plt.plot(X_org[idx, 0], X_org[idx, 1], COLORSX[k], label=group[k]);

                if (lmbda >= 9500):

                    print("*** in plot_clusters_vs_lambda  ********")

                    cluster_sz = np.shape(X_org[idx, :])[0]
                    cluster_no = np.full(shape=cluster_sz,fill_value=k, dtype=np.int)
                    print(np.shape(X_org), np.shape(idx), np.shape(X_org[idx, :]), np.shape(demograph[idx]), np.shape(cluster_no))

                    X_overall = np.concatenate((X_org[idx, :], demograph[idx].reshape(-1, 1), cluster_no.reshape(-1, 1)), axis=1)
                    all_data.append(pd.DataFrame(X_overall, columns=['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'Sex', 'Cluster_No']))

            if(len(all_data) > 0):
                df = pd.concat(all_data)
                df.to_csv(path + "Clusters_" + cluster_option + "_" + str(len(all_data)) + ".csv", index=False)

            if(lmbda >= 9500):

                if dataset == 'Synthetic' or dataset == 'Sensor':
                    tmp_title = '$\lambda$ = {}, fairness Error = {: .2f}'.format(lmbda,fairness_error)
                else:
                     tmp_title = '$\lambda$ = {}, fairness Error = {: .2f}'.format(lmbda,fairness_error)
                plt.title(tmp_title)
                plt.legend()
                plt.tight_layout()
                plt.savefig(filename, format='png', bbox_inches='tight')

            #plt.show()
            #plt.close('all')

def plot_fairness_vs_clusterE(cluster_option, savefile, filename, lmbdas, fairness_error_set, min_balance_set, avg_balance_set, E_cluster_set, save = True):

        if not osp.exists(savefile) or save == True:
            np.savez(savefile, lmbdas = lmbdas, min_balance_set = min_balance_set, avg_balance_set = avg_balance_set, fairness_error = fairness_error_set, E_cluster = E_cluster_set)
        else:
            data = np.load(savefile)
            lmbdas = data['lmbdas']
            fairness_error_set = data['fairness_error']
            E_cluster_set = data['E_cluster']
        # pdb.set_trace()
        if cluster_option == 'kmeans':
            label_cluster = 'K-means'
        elif cluster_option == 'ncut':
            label_cluster = 'Ncut'
        elif cluster_option == 'kmedian':
            label_cluster = 'K-medians'
            
        dataset = (filename.split('_')[-1].split('.'))[0]
        
        title = '{} Dataset ---- Fair {}'.format(dataset,label_cluster)

        ylabel1 = 'Fairness error'
        ylabel2 = '{} discrete energy'.format(label_cluster)


        length = len(lmbdas)
        plt.ion()
        fig, ax1 = plt.subplots()
        # ax1.set_xlim ([0,length])
        ax2 = ax1.twinx()

        ax1.plot(lmbdas[:length], fairness_error_set[:length], '--rD' , linewidth=2.5, label = ylabel1)

        ax2.plot(lmbdas[:length], E_cluster_set[:length], '--bP' , linewidth=3, label = ylabel2)
        ax1.set_xlabel(r'$\lambda$')
        ax1.set_ylabel(ylabel1, color = 'r')
        ax2.set_ylabel(ylabel2,color = 'b')
        ax1.legend(loc = 'upper right', bbox_to_anchor=(1, 0.6))
        ax2.legend(loc = 'upper right', bbox_to_anchor=(1, 0.7))

        fig.suptitle(title)
        fig.savefig(filename, format='png', dpi = 800, bbox_inches='tight')
        plt.show()
        plt.close('all')

def plot_K_vs_clusterE(cluster_option, savefile, filename, K_list, E_cluster_set, E_0_cluster_set, save = True):

        if not osp.exists(savefile) or save == True:
            np.savez(savefile, K_list = K_list, E_cluster_set = E_cluster_set, E_0_cluster_set = E_0_cluster_set)
        else:
            data = np.load(savefile)
            K_list = data['K_list']
            E_cluster_set = data['E_cluster_set']
            E_0_cluster_set = data['E_0_cluster_set']
        # pdb.set_trace()
        if cluster_option == 'kmeans':
            label_cluster = 'K-means'
        elif cluster_option == 'ncut':
            label_cluster = 'Ncut'
        elif cluster_option == 'kmedian':
            label_cluster = 'K-medians'

        dataset = (filename.split('_')[-1].split('.'))[0]

        title = '{} Dataset ---- Fair {}'.format(dataset,label_cluster)

        ylabel = '{} discrete energy'.format(label_cluster)

        legend_1 = 'Variational Fair {}'.format(label_cluster)
        legend_2 = 'Vanilla {}'.format(label_cluster)

        plt.figure(1,figsize=(6.4,4.8))
        plt.ion()
        plt.clf()
        plt.plot(K_list, E_cluster_set, '--gD' , linewidth=2.2)
        plt.plot(K_list, E_0_cluster_set, '--bP' , linewidth=2.2)
        plt.xlabel('Number of clusters (K)')
        plt.ylabel(ylabel)
        plt.legend([legend_1, legend_2], loc='upper center')
        plt.savefig(filename, format='png', dpi = 800, bbox_inches='tight')
        plt.show()
        plt.close('all')


def plot_balance_vs_clusterE(cluster_option, savefile, filename, lmbdas, fairness_error_set, min_balance_set, avg_balance_set, E_cluster_set, save = True):


        if not osp.exists(savefile) or save == True:
            np.savez(savefile, lmbdas = lmbdas, fairness_error = fairness_error_set, min_balance_set = min_balance_set, avg_balance_set = avg_balance_set, E_cluster = E_cluster_set)
        else:
            data = np.load(savefile)
            lmbdas = data['lmbdas']
            avg_balance_set = data['avg_balance_set']
            E_cluster_set = data['E_cluster']
        # pdb.set_trace()
        if cluster_option == 'kmeans':
            label_cluster = 'K-means'
        elif cluster_option == 'ncut':
            label_cluster = 'Ncut'
        elif cluster_option == 'kmedian':
            label_cluster = 'K-medians'

        dataset = (filename.split('_')[-1].split('.'))[0]

        title = '{} Dataset ---- Fair {}'.format(dataset,label_cluster)

        ylabel1 = ' Average Balance'
        ylabel2 = '{} discrete energy'.format(label_cluster)


        length = len(lmbdas)
        plt.ion()
        fig, ax1 = plt.subplots()
        # ax1.set_xlim ([0,length])
        ax2 = ax1.twinx()

        ax1.plot(lmbdas[:length], avg_balance_set[:length], '--rD' , linewidth=2.5, label = ylabel1)

        ax2.plot(lmbdas[:length], E_cluster_set[:length], '--bP' , linewidth=3, label = ylabel2)
        ax1.set_xlabel(r'$\lambda$')
        ax1.set_ylabel(ylabel1, color = 'r')
        ax2.set_ylabel(ylabel2,color = 'b')
        ax1.legend(loc = 'upper right', bbox_to_anchor=(1, 0.6))
        ax2.legend(loc = 'upper right', bbox_to_anchor=(1, 0.7))

        fig.suptitle(title)
        fig.savefig(filename, format='png', dpi = 800, bbox_inches='tight')
        plt.show()
        plt.close('all')


def plot_convergence(cluster_option, filename, E_fair):
    
    # Plot original fair clustering energy
    
    if cluster_option == 'kmeans':
        label_cluster = 'K-means'
    elif cluster_option == 'ncut':
        label_cluster = 'Ncut'
    elif cluster_option == 'kmedian':
        label_cluster = 'K-medians'
        
    length = len(E_fair)
    iter_range  = list(range(1,length+1))
    plt.figure(1,figsize=(6.4,4.8))
    plt.ion()
    plt.clf()
    ylabel = 'Fair {} objective'.format(label_cluster)
    plt.plot(iter_range, E_fair, 'r-' , linewidth=2.2)
    plt.xlabel('outer iterations')
    plt.ylabel(ylabel)
    plt.xlim(1,length)
    plt.savefig(filename, format='png', dpi = 800, bbox_inches='tight')
    plt.show()
    plt.close('all')
    
if __name__ == '__main__':
    cluster_option = 'ncut'
    data_dir = 'data/Sensor'
    dataset = 'fair_sensor.csv'
    output_path = 'outputs'
    dir_path = osp.join(data_dir, cluster_option + "_" + str(4))

    savefile = osp.join(dir_path,'Fair_{}_K_vs_clusterEdiscrete_{}.npz'.format(cluster_option,dataset))
    filename = osp.join(dir_path,'Fair_{}_K_vs_clusterEdiscrete_{}.png'.format(cluster_option,dataset))
    plot_K_vs_clusterE(cluster_option, savefile, filename, [], [], [], save=True)
    #
    # savefile = osp.join(data_dir,'Fair_{}_fairness_vs_clusterEdiscrete_{}.npz'.format(cluster_option,dataset))
    # filename = osp.join(data_dir,'Fair_{}_fairness_vs_clusterEdiscrete_{}.png'.format(cluster_option,dataset))
    # plot_fairness_vs_clusterE(cluster_option, savefile, filename, [], [], [], [], [], save = False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    