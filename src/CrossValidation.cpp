#include "utils/utils.h"

int main() {
    cout << "INFO - Execution started" << endl;
    // Variable to quantify the execution time
    auto start_time = chrono::high_resolution_clock::now();
    
    // Number of folds to perform the cross validation
    int K_folds = 10;

    // Model hyper-parameters
    int delta_time = 1800; // in seconds
    int delta_space = 150; // in meters
    double alpha = 0.1;
    double max_edge_distance = 1000; // in meters

    // Grid boundaries definition
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
    int max_time = 3600 * 24 * 7 * 4; // Highest timestamp in the dataset

    int T = (double(max_time)/(time_scale*delta_time) + 1);
    int M = ceil(double(max_longitude)/delta_space);
    int N = ceil(double(max_latitude)/delta_space);

    vector<vector<vector<int>>> graph (M*N, vector<vector<int>> (T, vector<int> ()));
    vector<int> node_samples (M*N, 0);
    for (int i = 0; i < num_samples; i++) {
        int t = data[0][i]/(time_scale*delta_time);
        int m = data[7][i]/delta_space;
        int n = data[6][i]/delta_space;

        if (t < T and m < M and n < N) {
            graph[N*m+n][t].push_back(i);
            node_samples[N*m+n]++;
        }
    }

    // Count the nodes without any samples (locations without measurements)
    int nodes_without_samples = 0;
    for (int j = 0; j < N*M; j++) {
        if (node_samples[j] == 0) {
            nodes_without_samples++;
        }
    }

    int nodes = N*M - nodes_without_samples;
    
    // Copy the graph before removing nodes without samples
    vector<vector<vector<int>>> graph_copy = graph;

    for (int j = M*N - 1; j >= 0; j--) {
        if (node_samples[j] == 0) {
            graph.erase(graph.begin() + j);
        }
    }

    // Variables to keep each folds results
    int test_fold;
    vector<double> test_results (K_folds);
    vector<double> test_std (K_folds, 0);

    // Cross validation loop
    for (int q = 0; q < K_folds; q++) {
        test_fold = q;
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
            // Traverse only the lower triangular part of the matrix (symmetric)
            for (int j = i + 1; j < nodes; j++) {
                // Compute distance between nodes
                double n1 = A[i].first % N;
                double n2 = A[j].first % N;
                double m1 = (A[i].first - n1)/N;
                double m2 = (A[j].first - n1)/N;
                double dist_nodes = double(delta_space) * sqrt((n1-n2)*(n1-n2) + (m1-m2)*(m1-m2));
                // Only compute the weight if the distance is smaller than max_edge_dist
                if (dist_nodes <= max_edge_distance) {
                    double weight = 0;
                    int count = 0;
                    // Time loop (one iteration per time interval)
                    for (int k = 0; k < T; k++) {
                        int nn1 = graph[i][k].size();
                        int nn2 = graph[j][k].size();
                        // Compute squared differences for every possible combination
                        for (int l1 = 0; l1 < nn1; l1++) {
                            for (int l2 = 0; l2 < nn2; l2++) {
                                int node_i = graph[i][k][l1];
                                int node_j = graph[j][k][l2];
                                // Check that the observation does not belong to the test fold
                                if (node_i % K_folds != test_fold and node_j % K_folds != test_fold) {
                                    // Update weight and count
                                    weight += (data[1][node_i]-data[1][node_j])*(data[1][node_i]-data[1][node_j]);
                                    count++;
                                }
                            }
                        }
                    }
                    if (count >= 1) {
                        // Apply the alpha factor of the formula and take the average
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

        // Map the correspondence between nodes before removing the not visited ones
        map<int,int> correspondence, correspondence_back;
        for (int i = 0; i < A.size(); i++) {
            correspondence[A[i].first] = i;
            correspondence_back[i] = A[i].first;
        }

        // Prepare the adjacency matrix for the maximum spanning tree algorithm
        vector<vector<double>> Adj = vector<vector<double>> (A.size(), vector<double> (A.size(), 0));
        for (int i = 0; i < A.size(); i++) {
            for (int j = i + 1; j < A.size(); j++) {
                if (A[i].second[j] != 0) {
                    Adj[i][j] = A[i].second[j];
                    Adj[j][i] = A[i].second[j];
                }
            }
        }
        
        // Graph density computation
        double graph_nodes = Adj.size();
        double max_graph_edges = graph_nodes*(graph_nodes-1)/2;
        double graph_edges = 0;
        for (int i = 0; i < graph_nodes; i++)
            for (int j = i + 1; j < graph_nodes; j++)
                if (Adj[i][j] != 0) graph_edges++;

        //cout << "Graph density: " << graph_edges/(max_graph_edges) << endl;

        // Compute the maximum spanning tree and save it as an intermediate result
        maximum_spanning_tree(Adj, "TreeAdjacencyMatrix.csv");

        // Load the tree graph
        vector<vector<double>> tree = read_CSV("TreeAdjacencyMatrix.csv");

        // Map the correspondence back between nodes from 1 to N to their corresponding labeled positions in space
        // Store this graph in a new file (where the mapping is clear)
        ofstream myFile("TreeGraph.csv");
        myFile << "node1,node2,weight,corresp1,corresp2" << endl;
        for (int i = 0; i < tree[0].size(); i++) {
            myFile << tree[0][i] << ',' << tree[1][i] << ',' << tree[2][i] << ',' <<
                        correspondence_back[tree[0][i]-1]+1 << ',' << correspondence_back[tree[1][i]-1]+1 << endl;
        }
        myFile.close();

        // Build the adjacency matrix for the tree graph
        vector<vector<double>> Adj_Tree = vector<vector<double>> (A.size(), vector<double> (A.size(), 0));
        for (int i = 0; i < tree[0].size(); i++) {
            Adj_Tree[tree[0][i]-1][tree[1][i]-1] = tree[2][i];    
            Adj_Tree[tree[1][i]-1][tree[0][i]-1] = tree[2][i];    
        }

        vector<int> interpolated_samples;
        double gsp_test_error = 0;
        int gsp_test_count = 0;
        vector<double> se;
        
        // Sample interpolation
        for (int i = test_fold; i < num_samples; i += K_folds) {
            int t = data[0][i]/(delta_time*time_scale); // Time frame
            // Identify the corresponding graph node
            int m = data[7][i]/delta_space;
            int n = data[6][i]/delta_space;
            // Check that the observation belongs to a valid time interval and node
            if (t < T and m < M and n < N) {
                double interpolated_value = 0;
                double total_weight = 0;
                int node = correspondence[N*m+n];
                bool found = false, anyFound = false;
                int hops = 0; // number of hops (k-hops)
                // Define a variable to store the traversed nodes so far in each possible path:
                // each row is a path, and each column contains a visited node
                vector<vector<int>> paths;
                paths.push_back({node});
                // Variable to store the k-hop paths that allow a valid interpolation
                vector<vector<int>> good_paths;
                while (not found) {
                    anyFound = false;
                    // Find a connection
                    int num_iter = paths.size();
                    
                    for (int p = 0; p < num_iter; p++) {
                        // Check all possible paths of length hops + 1
                        if (hops+1 == paths[p].size()) {
                            node = paths[p][hops];
                            int prev = -1; // previous node
                            if (hops > 0) {
                                prev = paths[p][hops-1];
                            }
                            for (int j = 0; j < Adj_Tree.size(); j++) {
                                // If there is an edge and the node is not the previous, we found
                                // a new node in the path
                                if (Adj_Tree[node][j] != 0 and j != prev) {
                                    // If it contains auxiliar observations, then we can make a valid interpolation
                                    if(graph_copy[correspondence_back[j]][t].size() != 0) {
                                        found = true;
                                        anyFound = true;
                                        // Take the path and mark it as a valid one for interpolation
                                        vector<int> path_copy = paths[p];
                                        path_copy.push_back(j);
                                        good_paths.push_back(path_copy);
                                    } else {
                                        // If there is an edge but there are no observations to interpolate, update
                                        // the path and keep looking for a node with valid interpolations
                                        vector<int> path_copy = paths[p];
                                        path_copy.push_back(j);
                                        paths.push_back(path_copy);
                                        // Set anyFound to true, since we found a possible path that does not have a dead end
                                        anyFound = true;
                                    }
                                }
                            }
                        }
                    }
                    hops++; // Increase hops variable
                    // If no paths were found, we reached a dead end, so we exit the loop.
                    // This is very unlikely to happen, only happens when the origin node is the only one in the graph
                    // containing observations in that time interval.
                    if (not anyFound) {
                        found = true;
                    }
                }
                if (anyFound) {
                    // Traverse the valid paths for interpolation
                    for (int j = 0; j < good_paths.size(); j++) {
                        // Compute each path's weight and the predicted value
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
        
        gsp_test_error /= gsp_test_count;
        double gsp_test_rmse = sqrt(gsp_test_error);
        cout << "Test Fold " << test_fold << " Results:" << endl << "RMSE = " << gsp_test_rmse << endl;
        test_results[test_fold] = gsp_test_rmse;        
    }

    double mean_test_rmse = 0;
    double mean_test_std_dev = 0;
    for (int i = 0; i < K_folds; i++) {
        mean_test_rmse += test_results[i];
    }
    mean_test_rmse /= K_folds;
    for (int i = 0; i < K_folds; i++) {
        mean_test_std_dev += (test_results[i]-mean_test_rmse) * (test_results[i]-mean_test_rmse);
    }
    mean_test_std_dev /= (K_folds-1);
    
    mean_test_std_dev = sqrt(mean_test_std_dev);

    cout << "FINAL RESULTS:\nMean test RMSE = " << mean_test_rmse << endl;

    auto end_time = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "Execution time: " << double(duration.count())/1000.0 <<  " seconds" << endl;
}