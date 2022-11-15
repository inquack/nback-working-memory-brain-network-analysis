import pickle
import numpy as np
import networkx as nx
import pandas as pd
import operator
import scipy.stats
import math
import matplotlib.pyplot as plt
#from brainx import util, detect_modules, modularity

######################################################
#
#  Methods for building keycodes from workspace files
#
#######################################################
def build_key_codes_from_workspaces(workspaces, datadir):
    """Return a list of unique Brainmap ids from a list of workspace files. 
    Function reads the individual files and builds a unique key from column one and five
    requires pandas imported as pd 
    
    
    Parameters:
    ------------
    workspaces: a python list
                list of filenames for workspaces
    datadir: string
                location of where all the workspace csvs are
    
    
    Returns:
    ----------
    key_codes: a python nested list of numeric key codes representing brainmap study+contrast unique ids
               every index is the list of key codes in the corresponding workspace. 
               length of list = number of workspaces
               length of each sublist= number of keys in workspace
    
    """
    key_codes=[]
    for x in workspaces:
        file =datadir+x[:-1]
        df = pd.read_csv(file, names = ['zero','one','two', 'three','four','five','six','seven','eight', 'nine'])
        true_only= df[df['zero']>0]
        one_col = true_only['one']
        five_col = true_only['five']
        key_codes.append(one_col.map(str).values + five_col.map(str).values)
    return key_codes

def build_keycodes_from_excel_csv(excel_csv_file):
    df=pd.read_csv(excel_csv_file)
    regionlist=list(df.keys())
    relabel_dict={idx:x[:] for idx, x in enumerate(regionlist)}
    
    keycodes=[]
    for key in list(df.keys()):
        studies_in_region=df[key][pd.notnull(df[key])]
        nstudies=len(studies_in_region)
        regions_in_study_list=[]
        [regions_in_study_list.append(studies) for studies in studies_in_region]
        keycodes.append(regions_in_study_list)
    return keycodes

######################################################
#
#  Methods for creating, normalizing & thresholding
#  co-activation matrices from "keycode" dictionaries
#
#######################################################


##### Methods for building Co-Activation matrices
 
def build_jaccard(key_codes):
    array_dims=len(key_codes), len(key_codes)
    jaccard=np.zeros(shape=array_dims)
    for col_idx, x in enumerate(key_codes):
        for row_idx, y in enumerate(key_codes):
            intersect=float(len(set(x) & set(y)))
            union=float(len(list(set(x) | set(y))))
            if union:
                jaccard[col_idx, row_idx]=intersect/union
    return jaccard
    

def build_n_coactives_array(key_codes):
    array_dims=len(key_codes), len(key_codes)
    n_coactives_array =np.zeros(shape=array_dims)
    
    for col_idx, x in enumerate(key_codes):
        for row_idx, y in enumerate(key_codes):
            n_coactives_array[col_idx, row_idx]=len(set(x) & set(y))
    return n_coactives_array
    

###### Methods for thresholding Graphs

def applycost_to_g(G,cost):
    """Threshold graph to achieve cost.

    Return the graph after the given cost has been applied. Leaves weights intact.

    Parameters
    ----------
    G: input networkx Graph

    cost: float
        Fraction or proportion of all possible undirected edges desired in the
        output graph.


    Returns
    -------
    G_cost: networkx Graph
        Networkx Graph with cost applied

    threshold: float
        Correlations below this value have been set to 0 in
        thresholded_corr_mat."""
        
    Complete_G=nx.complete_graph(G.number_of_nodes())
    n_edges_keep=int(Complete_G.number_of_edges()*cost)
    weights=[(G[x][y]['weight']) for x,y in G.edges_iter()]
    sorted_weights=sorted(weights, reverse=1)
    thresh=sorted_weights[n_edges_keep+1]
    remove_edgelist=[(x,y) for x,y in G.edges_iter() if G[x][y]['weight']<thresh]
    G_cost=G.copy()
    G_cost.remove_edges_from(remove_edgelist)
    return G_cost

def remove_edges_by_weight(G, max_weight):
    G_removed=G.copy()
    for x,y in G.edges_iter():
        if G_removed[x][y]['weight']<max_weight:
            G_removed.remove_edge(x,y)
            
    for x,y in G.degree_iter():
        if y<2:
            G_removed.remove_node(x)
    return G_removed 


def significant_connection_threshold(in_array,total_contrasts,threshold):
    """Function thresholds the array to keep edges that are statistically significant
    
    Parameters:
    -----------
    in_array: coactivation matrix
    array_lenght: dimension of the array
    total_contrasts: total number of contrasts reported by all papers taken from database
    threshold: threshold for significance
    
    Return:
    -------
    Returns thresh_array with significant connections kept and nonsignificant connections made to 0"""
    
    thresh_array=in_array.copy()
    for x in range(np.shape(thresh_array)[0]):
        for y in range(np.shape(thresh_array)[0]):
            if x != y: 
                # p=m/N m=independent activations of region X; N=total number of contrasts
                p=thresh_array[x][x]/total_contrasts 
                # null hypothesis is Binomial distribution of (k;n,p)* Binomial distribution of (m-k;N-n,p)
                # k=Coactivation of region X & Y, n=independent activation of region Y
                null=(scipy.stats.binom.pmf(thresh_array[x][y], thresh_array[y][y], p))*(scipy.stats.binom.pmf((thresh_array[x][x]-thresh_array[x][y]), (total_contrasts-thresh_array[y][y]), p))
                # dependence between activations between both regions defined by p_one and p_zero
                p_one=(thresh_array[x][y])/(thresh_array[y][y]) 
                p_zero=(thresh_array[x][x]-thresh_array[x][y])/(total_contrasts-thresh_array[y][y])
                # likelihood regions are functionally connected
                alternate=(scipy.stats.binom.pmf(thresh_array[x][y], thresh_array[y][y], p_one))*(scipy.stats.binom.pmf((thresh_array[x][x]-thresh_array[x][y]), (total_contrasts-thresh_array[y][y]), p_zero))
                # calculation of p value
                p_val=(-2*(math.log10(null/alternate)))
                # setting connection between region X and Y to zero if insignificant
                if p_val > threshold:
                    thresh_array[x][y]=0
    return thresh_array



# Methods related to computing influence or dependence b/w two nodes   
def build_influence_matrix(n_coactives_array):
    n_coactive_mat=n_coactives_array.copy()
    diagonal=n_coactive_mat.diagonal()
    a=n_coactive_mat/diagonal[:, np.newaxis] #dividing by row
    b=n_coactive_mat/diagonal #dividing by column
    influence_mat=a-b # positive: rows influence column (A infl. B) , negative: col influence row (B inf. A) --> only in the upper triangle 
    #influence_mat=np.triu(influence_mat)
    return influence_mat
    
def build_influence_digraph(n_coactives_array):
    influence_mat=build_influence_matrix(n_coactives_array)
    influence_di_mat=influence_mat*(influence_mat>0)
    influence_diG=nx.DiGraph(influence_di_mat)
    return influence_diG

def build_positive_influence_matrix(n_coactives_array):
    influence_mat=build_influence_matrix(n_coactives_array)
    influence_di_mat=influence_mat*(influence_mat>0)
    return influence_di_mat

##################################################
#
# Utility Methods
#
#################################################
 

def build_region_labels_dict(regionlist, trim=5):
    relabel_dict ={idx:x[:-trim] for idx, x in enumerate(regionlist)}
    return relabel_dict


def make_brainx_style_partition(community_part_dict):
    bx_part=[]
    for x in list(set(community_part_dict.values())):
        sub_part=[y for y in list(community_part_dict.keys()) if community_part_dict[y]==x ]
        #I could make sub_part a set
        bx_part.append(sub_part)
    return bx_part
    


def Z_transform_mat(M):
    indices_l=list(zip(np.tril_indices(len(M), k=-1)[0], np.tril_indices(len(M),k=-1)[1]))
    indices_diag=np.diag_indices(len(M))
    V=[M[y] for x,y in enumerate(indices_l)]
    Vm=np.mean(V)
    Vstd=np.std(V)
    Z=(M-Vm)/Vstd
    Z[indices_diag]=0
    return Z




##################################################
#
# Methods for making control graphs from BrainMap 
#
#################################################

def select_n_random_keycodes(keycodes, nstudies):
    unique_keycodes=[]
    for x in keycodes:
        unique_keycodes.extend(x)
    
    unique_keycodes=set(unique_keycodes)
    rands=np.random.rand(len(unique_keycodes))
    random_keys_tuple=list(zip(rands, unique_keycodes))
    selected_tuple=sorted(random_keys_tuple, reverse=1)[0:nstudies]
    selected_keys=list(zip(*selected_tuple))[1]
    selected_keycodes=[set(selected_keys) & set(x) for x in keycodes]
    return selected_keycodes

def build_average_graph_from_random_keycodes(keycodes, nstudies, niters):
    jaccards=np.ndarray([len(keycodes),len(keycodes), niters])
    for x in range(niters):
        select_n_keys=select_n_random_keycodes(keycodes, nstudies)
        jaccard=build_jaccard(build_n_coactives_array(select_n_keys))
        jaccards[:,:,x]=jaccard
    G=nx.from_numpy_matrix(np.mean(jaccards, 2))
    return G



def number_of_contrasts(key_codes):
    contrast_list=[]
    for x in key_codes:
        contrast_list.extend(x)
    ncontrasts=len(set(contrast_list))
    return ncontrasts
     




##################################################
#
# Methods for working with and computing metrics for
# Networkx Graphs 
#
#################################################

def remove_weight_edge_attribute(G):
    G_binary=G.copy()
    for x,y in G_binary.edges_iter():
        del G_binary[x][y]['weight']
    return G_binary


def remove_edgeless_nodes(G):
    to_remove=[]
    degree = G.degree()
    for x in degree:
        if degree(x)==0:
            to_remove.append(x)

    G.remove_nodes_from(to_remove)
    return G
    
    

def build_binarized_graph(G):
    """Takes graph converts it to a np array, binarizes it, builds networkx graph with binary edges"""
    
    binary_array=nx.to_numpy_matrix(G)
    binary_array=np.where(binary_array>0, 1,0)
    binary_G=nx.from_numpy_matrix(binary_array)
    #if nx.is_connected(binary_G)==0:
     #   print "Graph is not connected, removing nodes"
      #  binary_G=_remove_edgeless_nodes(binary_G)
    #else:
#        print "Graph is connected"
    return binary_G
 
 
 
def run_basic_metrics(G, top_n=5):
     """runs a bunch of basic metrics and returns a dict"""
     basic_metrics=dict()
     basic_metrics['degrees']=nx.degree(G)
     basic_metrics['cpl']=nx.average_shortest_path_length(G) #removed correction for disconnected components 
     basic_metrics['ccoeff']=nx.clustering(G)
     basic_metrics['degree_cent']=nx.degree_centrality(G)
     basic_metrics['between_cent']=nx.betweenness_centrality(G)
     for x in ['degree_cent','between_cent','ccoeff']:
         sorted_x = sorted(list(basic_metrics[x].items()), key=lambda item: item[1])
         tops=[]
         for y in range(top_n):
             tops.append(sorted_x[-top_n:][y])
         tops.reverse()
         basic_metrics['top'+x]=tops
         for x in ['degrees']:
             sorted_x = sorted(list(basic_metrics[x]), key=lambda item: item[1])
             tops=[]
         for y in range(top_n):
            tops.append(sorted_x[-top_n:][y])
         tops.reverse()
         basic_metrics['top'+x]=tops
    
     return basic_metrics

def run_weighted_metrics(G, top_n=5):
     """runs a bunch of basic metrics and returns a dict"""
     weighted_metrics=dict()
     weighted_metrics['degrees']=nx.degree(G, weight='weight')
     weighted_metrics['cpl']=nx.average_shortest_path_length(G, weight='weight') #removed correction for disconnected components
     weighted_metrics['ccoeff']=nx.clustering(G, weight='weight')
     weighted_metrics['degree_cent']=nx.degree_centrality(G)
     weighted_metrics['between_cent']=nx.betweenness_centrality(G, weight='weight')
     for x in ['degree_cent','between_cent','ccoeff']:
         sorted_x = sorted(list(weighted_metrics[x].items()), key=lambda item: item[1])
         tops=[]
         for y in range(top_n):
             tops.append(sorted_x[-top_n:][y])
         tops.reverse()
         weighted_metrics['top'+x]=tops
     for x in ['degrees']:
         sorted_x = sorted(list(weighted_metrics[x]), key=lambda item: item[1])
         tops=[]
         for y in range(top_n):
             tops.append(sorted_x[-top_n:][y])
         tops.reverse()
         weighted_metrics['top'+x]=tops
    
     return weighted_metrics
    
    

##################################################
#
# Methods for filtering of BrainMap keycode data
#
#################################################

def domain_filter_keycodes(key_codes, studies_filtered_by_domain, domain):
    """ filters a keycodes list by behavioral domain
        
        Parameters
        ------------
    
        key_codes: list of lists
            list of key_codes lists by region
    
        studies_filtered_by_domain: dict
            keys = behavioral domain, values are key_codes that correspond to that behavioral domain 
        
        domain: selected behavioral domain string
        Includes: 'Memory', 'Working Memory', 'Emotion', 'Attention', 'Language', 'Vision', 'Audition'

    Returns
    ------------
    domain_filtered_codes: list of lists
                    list of key_codes lists by region filtered by a particular domain
    
    """
    domainlist=studies_filtered_by_domain[domain]
    domain_filtered_codes=[set(x) & set(domainlist) for x in key_codes]
    return domain_filtered_codes
    

    


##################################################
#
# Plotting Methods 
#
#################################################
def plot_pretty_adj_matrix(G, nodelist, tick_space=5, colormap='Spectral'):
    mat=nx.to_numpy_matrix(G, nodelist)
    tick_range=list(range(0,G.number_of_nodes(), tick_space))
    selected_labels=[nodelist[y] for x,y in enumerate(tick_range)]
    
    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    a.imshow(mat, interpolation='nearest', cmap=plt.get_cmap(colormap))
    a.set_xticks(tick_range)
    a.set_xticklabels(selected_labels, rotation='vertical')
    a.set_yticks(tick_range)
    a.set_yticklabels(selected_labels)
    fig.colorbar(a.imshow(mat, interpolation='nearest'))
    return a
    
def plot_weight_histogram(G):
    histo=[G[x][y]['weight'] for x,y in G.edges()]
    return plt.hist(histo)

    
