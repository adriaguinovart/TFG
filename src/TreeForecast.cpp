#include "utils/utils.h"

int main() {
    cout << "INFO - Execution started" << endl;
    // Variable to quantify the execution time
    auto start_time = chrono::high_resolution_clock::now();
    
    // Model hyper-parameters
    int delta_time = 3600 / 2;
    int delta_space = 150;
    double alpha = 0.1;
    double max_edge_distance = 1000;

    // Grid boundaries definition (in meters)
    int max_longitude = 9000;
    int max_latitude = 12000;

    // Setup to adapt the execution to different data timestamps resolution
    string time_resolution = "seconds";
    int time_scale;
    if (time_resolution == "milliseconds") {
        time_scale = 1000;
    } else if (time_resolution == "seconds") {
        time_scale = 1;
    } else {
        cout << "Unrecognized time scale" << endl;
    }

    // Loading dataset
    string path = "data/";
    string file = "processedData.csv";
    vector<vector<double>> data = read_CSV(path + file);

    int num_samples = data[0].size(); // Number of rows in the dataset
    long long max_time = data[0][num_samples-1]; // Highest timestamp in the dataset

    int T = (double(max_time)/(time_scale*delta_time) + 1);
    int T_week = (double(3600 * 24 * 7)/delta_time + 1);
    int M = ceil(double(max_longitude)/delta_space);
    int N = ceil(double(max_latitude)/delta_space);

    vector<vector<vector<int>>> graph (M*N, vector<vector<int>> (T, vector<int> ()));
    vector<int> node_samples (M*N, 0);
    for (int i = 0; i < num_samples; i++) {
        int t = data[0][i]/(time_scale*delta_time);
        int m = data[7][i]/delta_space;
        int n = data[6][i]/delta_space;

        if (t < 13*T_week and m < M and n < N) {
            graph[N*m+n][t].push_back(i);
            node_samples[N*m+n]++;
        } 
        else if (t < T and m < M and n < N) {
            graph[N*m+n][t].push_back(i);
        }
    }

    vector<vector<vector<int>>> graph_copy = graph;

    int nodes_without_samples = 0;

    for (int j = 0; j < N*M; j++) {
        if (node_samples[j] == 0) {
            nodes_without_samples++;
        }
    }

    int nodes = N*M - nodes_without_samples;
    
    for (int j = M*N - 1; j >= 0; j--) {
        if (node_samples[j] == 0) {
            graph.erase(graph.begin() + j);
        }
    }
    
    vector<pair<int,vector<double>>> A (nodes);
    vector<vector<int>> counts (nodes, vector<int> (nodes, 0));
    int pos = 0;

    for (int i = 0; i < nodes; i++) {

        while (node_samples[pos] == 0) {
            pos++;
        }
        pair<int,vector<double>> aux = {pos, vector<double> (nodes, 0)};
        A[i] = aux;
        pos++;
    }

    // Computation of weights
    vector<bool> visited(nodes, false);
    for (int i = 0; i < nodes; i++) {
        for (int j = i + 1; j < nodes; j++) {
            double n1 = A[i].first % N;
            double n2 = A[j].first % N;
            double m1 = (A[i].first - n1)/N;
            double m2 = (A[j].first - n1)/N;
            double dist_nodes = double(delta_space) * sqrt((n1-n2)*(n1-n2) + (m1-m2)*(m1-m2));
            if (dist_nodes <= max_edge_distance) {
                double weight = 0;
                int count = 0;
                for (int k = 0; k < 2*T_week; k++) {
                    int nn1 = graph[i][k].size();
                    int nn2 = graph[j][k].size();
                    for (int l1 = 0; l1 < nn1; l1++) {
                        for (int l2 = 0; l2 < nn2; l2++) {
                            int node_i = graph[i][k][l1];
                            int node_j = graph[j][k][l2];
                            weight += (data[1][node_i]-data[1][node_j])*(data[1][node_i]-data[1][node_j]);
                            count++;
                        }
                    }
                }
                if (count >= 1) {
                    weight = (weight + 2*alpha)/count;
                    weight = 1/weight;
                    A[i].second[j] = weight;
                    A[j].second[i] = weight;
                    counts[i][j] = count;
                    counts[j][i] = count;
                    visited[i] = true;
                    visited[j] = true;
                }
            }
        }
    }

    for (int i = nodes - 1; i >= 0; i--) {
        if (visited[i] == false) {
            A.erase(A.begin() + i);
            counts.erase(counts.begin()+i);
            for (int j = 0; j < A.size(); j++) {
                A[j].second.erase(A[j].second.begin()+i);
                counts[j].erase(counts[j].begin()+i);
            } 
        }
    }

    map<int,int> correspondence, correspondence_back;

    for (int i = 0; i < A.size(); i++) {
        correspondence[A[i].first] = i;
        correspondence_back[i] = A[i].first;
    }

    vector<vector<double>> Adj = vector<vector<double>> (A.size(), vector<double> (A.size(), 0));
    for (int i = 0; i < A.size(); i++) {
        for (int j = i + 1; j < A.size(); j++) {
            if (A[i].second[j] != 0) {
                Adj[i][j] = A[i].second[j];
                Adj[j][i] = A[i].second[j];
            }
        }
    }
    
    double graph_nodes = Adj.size();
    double max_graph_edges = graph_nodes*(graph_nodes-1)/2;
    double graph_edges = 0;
    for (int i = 0; i < graph_nodes; i++)
        for (int j = i + 1; j < graph_nodes; j++)
            if (Adj[i][j] != 0) graph_edges++;


    // Find the maximum spanning tree and store intermediate results
    maximum_spanning_tree(Adj, "TreeAdjacencyMatrix.csv");

    // Read the tree adjacency matrix
    vector<vector<double>> tree = read_CSV("TreeAdjacencyMatrix.csv");

    ofstream myFile("TreeGraph.csv");
    myFile << "node1,node2,weight,corresp1,corresp2" << endl;
    for (int i = 0; i < tree[0].size(); i++) {
        myFile << tree[0][i] << ',' << tree[1][i] << ',' << tree[2][i] << ',' <<
                    correspondence_back[tree[0][i]-1]+1 << ',' << correspondence_back[tree[1][i]-1]+1 << endl;
    }
    myFile.close();

    vector<vector<double>> Adj_Tree = vector<vector<double>> (A.size(), vector<double> (A.size(), 0));

    for (int i = 0; i < tree[0].size(); i++) {
        Adj_Tree[tree[0][i]-1][tree[1][i]-1] = tree[2][i];    
        Adj_Tree[tree[1][i]-1][tree[0][i]-1] = tree[2][i];    
    }

    vector<int> interpolated_samples;
    double gsp_test_error = 0;
    int gsp_test_count = 0;
    int not_found = 0;
    vector<double> se;
    for (int i = 0; i < num_samples; i++) {

        int t = data[0][i]/(delta_time*time_scale);
        if (t >= 2*T_week and t < 4*T_week) {
            int m = data[7][i]/delta_space;
            int n = data[6][i]/delta_space;
            double interpolated_value = 0;
            double total_weight = 0;
            if (correspondence.find(N*m+n) != correspondence.end()) {
                int node = correspondence[N*m+n];
                bool found = false, anyFound = false;
                int hops = 0;
                vector<vector<int>> paths;
                vector<vector<int>> good_paths;
                paths.push_back({node});
                while (not found) {
                    anyFound = false;
                    // Find a connection
                    int num_iter = paths.size();
                    for (int p = 0; p < num_iter; p++) {
                        if (hops+1 == paths[p].size()) {
                            node = paths[p][hops];
                            int prev = -1;
                            if (hops > 0) {
                                prev = paths[p][hops-1];
                            }
                            for (int j = 0; j < Adj_Tree.size(); j++) {
                                if (Adj_Tree[node][j] != 0 and j != prev) {
                                    if(graph_copy[correspondence_back[j]][t].size() != 0) {
                                        found = true;
                                        anyFound = true;
                                        vector<int> path_copy = paths[p];
                                        path_copy.push_back(j);
                                        good_paths.push_back(path_copy);
                                    } else {
                                        vector<int> path_copy = paths[p];
                                        path_copy.push_back(j);
                                        paths.push_back(path_copy);
                                        anyFound = true;
                                    }
                                }
                            }
                        }
                    }
                    hops++;
                    if (not anyFound) {
                        found = true;
                    }
                }
                if (anyFound) {
                    for (int j = 0; j < good_paths.size(); j++) {
                        double combined_weight = 0;
                        for (int k = 0; k < hops; k++) {
                            int curr_node = good_paths[j][k];
                            int next_node = good_paths[j][k+1];
                            combined_weight += 1/Adj_Tree[curr_node][next_node];
                        }
                        total_weight += 1/combined_weight;
                        double mean = 0;
                        int last_node = good_paths[j][hops];
                        for (int k = 0; k < graph_copy[correspondence_back[last_node]][t].size(); k++) {
                            mean += data[1][graph_copy[correspondence_back[last_node]][t][k]];
                        }
                        mean /= graph_copy[correspondence_back[last_node]][t].size();
                        interpolated_value += mean / combined_weight;
                    }
                    for (int j = 0; j < Adj_Tree.size(); j++) {
                        if (Adj_Tree[node][j] != 0 and graph_copy[correspondence_back[j]][t].size() != 0) {
                            double mean = 0;
                            for (int k = 0; k < graph_copy[correspondence_back[j]][t].size(); k++) {
                                mean += data[1][graph_copy[correspondence_back[j]][t][k]];
                            }
                            mean /= graph_copy[correspondence_back[j]][t].size();
                            interpolated_value += mean * Adj_Tree[node][j];
                            total_weight += Adj_Tree[node][j];
                            
                        }
                    }
                    if (total_weight != 0) {
                        interpolated_value /= total_weight;
                        gsp_test_error += (interpolated_value - data[1][i])*(interpolated_value - data[1][i]);
                        gsp_test_count++;
                        interpolated_samples.push_back(i);
                        se.push_back((interpolated_value - data[1][i])*(interpolated_value - data[1][i]));
                    }
                }
            }
        }
    }
    
    gsp_test_error /= gsp_test_count;
    double gsp_test_rmse = sqrt(gsp_test_error);
    cout << "--- RESULTS ---" << endl << "RMSE = " << gsp_test_rmse << endl;

    double se_mean = 0;
    for (int i = 0; i < se.size(); i++){
        se_mean += se[i];
    }
    
    se_mean /= se.size();
    double mse_std = 0;
    for (int i = 0; i < se.size(); i++)
        mse_std += (se[i]-se_mean)*(se[i]-se_mean);
    mse_std /= se.size();

    auto end_time = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "Execution time: " << double(duration.count())/1000 <<  " seconds" << endl;
}