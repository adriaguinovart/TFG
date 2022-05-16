import matplotlib.pyplot as plt
import numpy as np


def plot(X,Y,xlabel,ylabel,log=False):
    plt.rc('font', family='serif')
    plt.rc('xtick',labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X, Y, 
        marker='o',
        markersize=3,
        color = 'darkblue',
        markerfacecolor='black',
        markeredgecolor='black',
        markeredgewidth=2,
        linestyle='-',
        linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(linestyle='--', linewidth=0.5)
    if (log):
        ax.set_xscale('log')
    plt.show()

def plot_comparison(X,T,K,I,xlabel,ylabel,error=False,T_e=None,K_e=None,I_e=None):
    plt.rc('font', family='serif')
    plt.rc('xtick',labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)    
    
    p3 = ax.plot(X, I,
        marker='o',
        markersize=3,
        color = 'lime',
        markerfacecolor='lime',
        markeredgecolor='lime',
        markeredgewidth=2,
        linestyle='-',
        linewidth=2)

    p2 = ax.plot(X, K,
        marker='o',
        markersize=3,
        color = 'orange',
        markerfacecolor='orange',
        markeredgecolor='orange',
        markeredgewidth=2,
        linestyle='-',
        linewidth=2)

    p1 = ax.plot(X, T,
        marker='o',
        markersize=3,
        color = 'darkblue',
        markerfacecolor='darkblue',
        markeredgecolor='darkblue',
        markeredgewidth=2,
        linestyle='-',
        linewidth=2)

    ax.legend(handles=[p1[0],p2[0],p3[0]],labels=['Tree Graph', 'Kriging', 'Inverse Distance Weighting'])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(linestyle='--', linewidth=0.5)

    if (error):
        ax.errorbar(X, I,
            yerr=I_e,
            color='lime',
            ls='--', 
            marker='', 
            capsize=3, 
            capthick=1, 
            ecolor='lime')
        ax.errorbar(X, K,
            yerr=K_e,
            color='orange',
            ls='--', 
            marker='', 
            capsize=3, 
            capthick=1, 
            ecolor='orange')
        ax.errorbar(X, T,
            yerr=T_e,
            color='darkblue',
            ls='--', 
            marker='', 
            capsize=3, 
            capthick=1, 
            ecolor='darkblue')
    plt.show()
    
if __name__ == "__main__":

    delta_t = True
    delta_s = True
    alpha = True
    max_edge_dist = True
    max_edge_dist_time = True
    comparison_rmse_general = True
    comparison_rmse_specific = True
    comparison_time = True
    comparison_mse = True

    if (delta_t):
        x = np.array([0.5,1,1.5,2,3,4,6,8,12])
        y = np.array([0.1593118,0.283134012,0.404665112,0.735002866,0.798998088,0.936388811,0.970727146,1.05059846,1.092393997])
        plot(x,y,'\u0394t (h)','RMSE (ppb)')

    if (delta_s):
        x = np.array([50,75,100,125,150,175,200,250,300,400,500])
        y = np.array([0.174972875,0.16120664,0.195750678,0.225318961,0.1593118,0.209036743,0.211602228,0.211704233,0.229275692,0.278415285,0.323222678])
        plot(x,y,'\u0394s (m)','RMSE (ppb)')

    if (alpha):
        x = np.array([0.001,0.005,0.01,0.05,0.1,0.2,0.5,1,5,10,50,100])
        y = np.array([0.169559749,0.167220902,0.165729546,0.163111888,0.1593118,0.195405566,0.271971662,0.288283751,0.299673888,0.291165607,0.297448914,0.30760043])
        plot(x,y,'\u03B1','RMSE (ppb)',log=True)

    if (max_edge_dist):
        x = np.array([700,800,900,1000,1500,2000,3000,4000,5000,7500,10000])
        y = np.array([0.185767,0.181935,0.170692,0.1593118,0.1593118,0.1593118,0.1593118,0.1593118,0.1593118,0.1593118,0.1593118])
        plot(x,y,'Maximum edge distance (m)','RMSE (ppb)')    

    if (max_edge_dist_time):
        x = np.array([700,800,900,1000,1500,2000,3000,4000,5000,7500,10000])
        y = np.array([0.754,0.913,1.047,1.206,2.123,3.226,4.941,6.796,9.124,11.936,12.652])
        plot(x,y,'Maximum edge distance (m)','Execution time (s)')    

    if (comparison_rmse_general):
        x = np.array([0.5,1,1.5,2,4,8])
        tree = np.array([0.1593118,0.180332,0.203118,0.287196,0.406345,0.656668])
        kriging = np.array([0.204112656,0.356693758,0.438091288,0.799901392,1.424039953,1.703767888])
        idw = np.array([0.282297,0.450892,0.499555,0.880187,1.53423,1.67352])
        plot_comparison(x,tree,kriging,idw,'\u0394t (h)','RMSE (ppb)')

    if (comparison_rmse_specific):
        x = np.array([30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120])
        tree = np.array([0.163118,0.1931,0.165935,0.185632,0.169029,0.180786,0.180332,0.183951,0.190306,0.193015,0.190196,0.189161,0.203118,0.194299,0.327208,0.23175,0.233031,0.317413,0.287196])
        kriging = np.array([0.204112656,0.272019794,0.279524294,0.308271856,0.370360367,0.319340922,0.356693758,0.353616303,0.368053811,0.427902716,0.432000727,0.442325582,0.438091288,0.457934784,0.63425692,0.687243147,0.81083961,0.950659742,0.799901392])
        idw = np.array([0.282297,0.296012,0.347666,0.376463,0.397217,0.393086,0.450892,0.438145,0.523595,0.498532,0.537618,0.559554,0.499555,0.712658,0.682036,0.726195,0.802113,0.777316,0.880187])
        plot_comparison(x,tree,kriging,idw,'\u0394t (min)','RMSE (ppb)')

    if(comparison_time):
        x = np.array([0.5,1,1.5,2,4,8])
        tree = np.array([12.464,11.818,9.03,8.421,7.724,7.431])
        kriging = np.array([5.77,6.59,7.49,8.83,13.62,24.07])
        idw = np.array([9.629,9.062,8.452,8.248,8.359,9.117])
        plot_comparison(x,tree,kriging,idw,'\u0394t (h)','Execution time (s)')

    if (comparison_mse):
        x = np.array([30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120])
        tree = np.array([0.026607482,0.03728761,0.027534424,0.034459239,0.028570803,0.032683578,0.033610622,0.03383797,0.036216374,0.03725479,0.036174518,0.035781884,0.04248463,0.037752101,0.107065347,0.053708063,0.054303447,0.100751013,0.082481542])
        kriging = np.array([0.041661976,0.073994768,0.078133831,0.095031537,0.137166802,0.101978624,0.127230437,0.12504449,0.135463608,0.183100735,0.186624628,0.195651921,0.191923977,0.209704267,0.40228184,0.472303143,0.657460873,0.903753946,0.639842237])
        idw = np.array([0.079691596,0.087623104,0.120871648,0.14172439,0.157781345,0.154516603,0.203303596,0.191971041,0.274151724,0.248534155,0.289033114,0.313100679,0.249555198,0.507881425,0.465173105,0.527359178,0.643385265,0.604220164,0.774729155])
        t_e = np.array([0.004598405,0.0036944143,0.00251908,0.020042222,0.005504733,0.008493997,0.01479286,0.01224143,0.017278476,0.015557951,0.018532938,0.014797677,0.039092967,0.018075089,0.04301202,0.035196025,0.053221163,0.035481932,0.043971707])
        k_e = np.array([0.016089812,0.069573947,0.063787046,0.0678285952,0.0719332481,0.074577841,0.0605257788,0.07200367,0.085055131,0.101809247,0.124998437,0.10660662,0.119881926,0.111687525,0.128160179,0.135566755,0.185116952,0.26751557,0.315375331])
        i_e = np.array([0.07356734,0.060543801,0.064410297,0.085963938,0.098143301,0.072472011,0.093871964,0.088833309,0.102561638,0.159775515,0.139494026,0.19633984,0.199183901,0.245193033,0.276653924,0.267962099,0.325344208,0.415519076,0.561791147])
        plot_comparison(x,tree,kriging,idw,'\u0394t (min)',"MSE (ppb²)",error=True,T_e=t_e,K_e=k_e,I_e=i_e)



    