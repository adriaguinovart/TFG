#include "utils/utils.h"

int main() {
    cout << "INFO - Execution started" << endl;
    // Variable to quantify the execution time
    auto start_time = chrono::high_resolution_clock::now();
    
    // Nested Cross Validation (NCV) parameters
    int K_folds = 10;

    // Model hyper-parameters
    vector<int> delta_time = {1800, 3600, 7200}; // delta_time in seconds
    int num_delta_times = delta_time.size();
    vector<int> delta_space = {50, 100, 150}; // delta_space in meters
    int num_delta_spaces = delta_space.size();
    vector<double> alpha = {0.01, 0.1, 1, 10}; // alpha parameter for weights computation
    int num_alphas = alpha.size();
    vector<double> max_edge_distance = {1000, 2000, 5000}; // Maximum distance (in meters) between nodes considered
    int num_max_dist = max_edge_distance.size();
    
    // Variables that define the grid where all the measurements are located
    int max_longitude_distance = 9000;
    int max_latitude_distance = 12000;

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

    // Variables containing the number of time frames and spatial divisions 
    //in each of the axis considered in the 2D plane.
    vector<int> T;
    for (int i = 0; i < num_delta_times; i++) {
         T.push_back((double(max_time)/(time_scale*delta_time[i]) + 1));
    }
    vector<int> M, N;
    for (int i = 0; i < num_delta_spaces; i++) {
        M.push_back(ceil(double(max_longitude_distance)/delta_space[i]));
        N.push_back(ceil(double(max_latitude_distance)/delta_space[i]));
    }
    
    // Variable representing the data classified by time and space
    vector<vector<vector<vector<int>>>> graph(num_delta_times * num_delta_spaces);
    for (int i = 0; i < num_delta_times; i++) {
        for (int j = 0; j < num_delta_spaces; j++) {
            graph[num_delta_spaces*i + j] = vector<vector<vector<int>>> (M[j]*N[j], vector<vector<int>> (T[i], vector<int> ()));
        }
    }

    vector<vector<int>> node_samples(num_delta_spaces);
    for (int i = 0; i < num_delta_spaces; i++) {
        node_samples[i] = vector<int> (M[i]*N[i],0);
    }
    // Loop over the dataset to classify each observation into their
    // corresponding time frame and spatial cell
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_delta_times; j++) {
            for (int k = 0; k < num_delta_spaces; k++) {
                int t = data[0][i]/(time_scale*delta_time[j]);
                int m = data[7][i]/delta_space[k];
                int n = data[6][i]/delta_space[k];

                if (t < T[j] and m < M[k] and n < N[k]) {
                    graph[num_delta_spaces*j+k][N[k]*m+n][t].push_back(i);
                    node_samples[k][N[k]*m+n]++;
                }
            }
        }
    }

    vector<int> nodes_without_samples (num_delta_spaces, 0);
    for (int i = 0; i < num_delta_spaces; i++) {
        for (int j = 0; j < N[i]*M[i]; j++) {
            if (node_samples[i][j] == 0) {
                nodes_without_samples[i]++;
            }
        }
    }
    vector<int> nodes(num_delta_spaces);
    for (int i = 0; i < num_delta_spaces; i++) {
        nodes[i] = N[i]*M[i] - nodes_without_samples[i];
        cout << "Delta space = " << delta_space[i] << ":\nPercentage of nodes with samples: " << float(nodes[i])/float(N[i]*M[i])*100.0 << "%" << endl;

    }


    for (int i = 0; i < num_delta_spaces; i++) {
        for (int j = M[i]*N[i] - 1; j >= 0; j--) {
            if (node_samples[i][j] == 0) {
                for (int k = num_delta_times - 1; k >= 0; k--) {
                    graph[num_delta_spaces*k+i].erase(graph[num_delta_spaces*k+i].begin() + j);
                }
            }
        }
    }

    vector<vector<vector<vector<int>>>> graph_copy(num_delta_times * num_delta_spaces);
    for (int i = 0; i < num_delta_times; i++) {
        for (int j = 0; j < num_delta_spaces; j++) {
            graph_copy[num_delta_spaces*i + j] = vector<vector<vector<int>>> (M[j]*N[j], vector<vector<int>> (T[i], vector<int> ()));
        }
    }
    
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_delta_times; j++) {
            for (int k = 0; k < num_delta_spaces; k++) {
                int t = data[0][i]/(time_scale*delta_time[j]);
                int m = data[7][i]/delta_space[k];
                int n = data[6][i]/delta_space[k];

                if (t < T[j] and m < M[k] and n < N[k]) {
                    graph_copy[num_delta_spaces*j+k][N[k]*m+n][t].push_back(i);
                    node_samples[k][N[k]*m+n]++;
                }
            }
        }
    }

    // Tree-graph interpolation
    int validation_fold, test_fold;
    vector<vector<vector<vector<int>>>> best_params (num_delta_times, vector<vector<vector<int>>> (num_delta_spaces, vector<vector<int>>(num_alphas, vector<int> (num_max_dist, 0))));
    vector<double> test_results (K_folds);
    for (int q = 0; q < K_folds; q++) {
        test_fold = q;
        
        vector<vector<vector<vector<vector<double>>>>> fold_results_validation (num_delta_times, vector<vector<vector<vector<double>>>> (num_delta_spaces, vector<vector<vector<double>>>(num_alphas, vector<vector<double>>(num_max_dist, vector<double>()))));
        
        for (int w = 0; w < K_folds - 1; w++) {
            if (w != q) {
                validation_fold = w;
            } else {
                validation_fold = K_folds - 1;
            }
            
            for (int a = 0; a < num_alphas; a++) {
                for (int d = 0; d < num_max_dist; d++) {
                    for (int ds = 0; ds < num_delta_spaces; ds++) {
                        for (int dt = 0; dt < num_delta_times; dt++) {
                            vector<pair<int,vector<double>>> A (nodes[ds]);
                            vector<vector<int>> counts (nodes[ds], vector<int> (nodes[ds], 0));
                            int pos = 0;

                            for (int i = 0; i < nodes[ds]; i++) {

                                while (node_samples[ds][pos] == 0) {
                                    pos++;
                                }
                                pair<int,vector<double>> aux = {pos, vector<double> (nodes[ds], 0)};
                                A[i] = aux;
                                pos++;
                            }

                            vector<bool> visited(nodes[ds], false);
                            for (int i = 0; i < nodes[ds]; i++) {
                                for (int j = i + 1; j < nodes[ds]; j++) {
                                    double n1 = A[i].first % N[ds];
                                    double n2 = A[j].first % N[ds];
                                    double m1 = (A[i].first - n1)/N[ds];
                                    double m2 = (A[j].first - n1)/N[ds];
                                    double dist_nodes = double(delta_space[ds]) * sqrt((n1-n2)*(n1-n2) + (m1-m2)*(m1-m2));
                                    if (dist_nodes <= max_edge_distance[d]) {
                                        double weight = 0;
                                        int count = 0;
                                        for (int k = 0; k < T[dt]; k++) {
                                            int nn1 = graph[num_delta_spaces*dt+ds][i][k].size();
                                            int nn2 = graph[num_delta_spaces*dt+ds][j][k].size();
                                            for (int l1 = 0; l1 < nn1; l1++) {
                                                for (int l2 = 0; l2 < nn2; l2++) {
                                                    int node_i = graph[num_delta_spaces*dt+ds][i][k][l1];
                                                    int node_j = graph[num_delta_spaces*dt+ds][j][k][l2];
                                                    if (node_i % K_folds != validation_fold and node_j % K_folds != validation_fold and
                                                        node_i % K_folds != test_fold and node_j % K_folds != test_fold) {
                                                        weight += (data[1][node_i]-data[1][node_j])*(data[1][node_i]-data[1][node_j]);
                                                        count++;
                                                    }
                                                }
                                            }
                                        }
                                        if (count >= 1) {
                                            weight = (weight + 2*alpha[a])/count;
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


                            for (int i = nodes[ds] - 1; i >= 0; i--) {
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

                            // cout << "Graph density: " << graph_edges/(max_graph_edges) << endl;

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
                            double gsp_error = 0;
                            int gsp_count = 0;
                            for (int i = validation_fold; i < num_samples; i += K_folds) {

                                int t = data[0][i]/(delta_time[dt]*time_scale);
                                int m = data[7][i]/delta_space[ds];
                                int n = data[6][i]/delta_space[ds];
                                double interpolated_value = 0;
                                double total_weight = 0;
                                int node = correspondence[N[ds]*m+n];
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
                                                    if(graph_copy[num_delta_spaces*dt+ds][correspondence_back[j]][t].size() != 0) {
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
                                        for (int k = 0; k < graph_copy[num_delta_spaces*dt+ds][correspondence_back[last_node]][t].size(); k++) {
                                            mean += data[1][graph_copy[num_delta_spaces*dt+ds][correspondence_back[last_node]][t][k]];
                                        }
                                        mean /= graph_copy[num_delta_spaces*dt+ds][correspondence_back[last_node]][t].size();
                                        interpolated_value += mean / combined_weight;
                                    }
                                
                                    if (total_weight != 0) {
                                        interpolated_value /= total_weight;
                                        gsp_error += (interpolated_value - data[1][i])*(interpolated_value - data[1][i]);
                                        gsp_count++;
                                        interpolated_samples.push_back(i);
                                    }
                                }
                            }
                            
                            gsp_error /= gsp_count;
                            double gsp_rmse = sqrt(gsp_error);
                            
                            cout << "Validation fold: " << validation_fold << "; GSP --> RMSE = " << gsp_rmse << endl;
                            fold_results_validation[dt][ds][a][d].push_back(gsp_rmse);
                        }
                    }
                }
            }
        }
                
        
        vector<vector<vector<vector<double>>>> average_validation_results (num_delta_times, vector<vector<vector<double>>> (num_delta_spaces, vector<vector<double>> (num_alphas, vector<double> (num_max_dist, INT_MAX))));
        for (int i = 0; i < num_delta_times; i++) {
            for (int j = 0; j < num_delta_spaces; j++) {
                for (int a = 0; a < num_alphas; a++) {
                    for (int d = 0; d < num_max_dist; d++) {
                        int num_results = fold_results_validation[i][j][a][d].size();
                        if (num_results != 0) {
                            double avg = 0;
                            for (int k = 0; k < num_results; k++) {
                                avg += fold_results_validation[i][j][a][d][k];        
                            }
                            average_validation_results[i][j][a][d] = avg/num_results;
                        }
                    }
                }
            }
        }
        for (int i = 0; i < num_delta_times; i++) {
            for (int j = 0; j < num_delta_spaces; j++) {
                for (int a = 0; a < num_alphas; a++) {
                    for (int d = 0; d < num_max_dist; d++) {
                        cout << "Delta time (s) = " << delta_time[i] << "; Delta space (m) = " << delta_space[j] << "; Alpha = " <<
                        alpha[a] << "; Max dist (m) = " << max_edge_distance[d] << "; RMSE = " << average_validation_results[i][j][a][d] << endl;
                    }
                }
            }
        }
        vector<int> best_indices = find_best_indices(average_validation_results);

        cout << "BEST PARAMETERS: " << endl << "Delta time (s) = " << delta_time[best_indices[0]] << endl <<
                "Delta space (m) = " << delta_space[best_indices[1]] << endl << "Alpha = " << alpha[best_indices[2]] << endl <<
                "Max dist (m) = " << max_edge_distance[best_indices[3]] << endl;

        best_params[best_indices[0]][best_indices[1]][best_indices[2]][best_indices[3]]++;
        
        vector<pair<int,vector<double>>> A (nodes[best_indices[1]]);
        vector<vector<int>> counts (nodes[best_indices[1]], vector<int> (nodes[best_indices[1]], 0));
        int pos = 0;

        for (int i = 0; i < nodes[best_indices[1]]; i++) {

            while (node_samples[best_indices[1]][pos] == 0) {
                pos++;
            }
            pair<int,vector<double>> aux = {pos, vector<double> (nodes[best_indices[1]], 0)};
            A[i] = aux;
            pos++;
        }

        vector<bool> visited(nodes[best_indices[1]], false);
        for (int i = 0; i < nodes[best_indices[1]]; i++) {
            for (int j = i + 1; j < nodes[best_indices[1]]; j++) {
                double n1 = A[i].first % N[best_indices[1]];
                double n2 = A[j].first % N[best_indices[1]];
                double m1 = (A[i].first - n1)/N[best_indices[1]];
                double m2 = (A[j].first - n1)/N[best_indices[1]];
                double dist_nodes = double(delta_space[best_indices[1]]) * sqrt((n1-n2)*(n1-n2) + (m1-m2)*(m1-m2));
                if (dist_nodes <= max_edge_distance[best_indices[3]]) {
                    double weight = 0;
                    int count = 0;
                    for (int k = 0; k < T[best_indices[0]]; k++) {
                        int nn1 = graph[num_delta_spaces*best_indices[0]+best_indices[1]][i][k].size();
                        int nn2 = graph[num_delta_spaces*best_indices[0]+best_indices[1]][j][k].size();
                        for (int l1 = 0; l1 < nn1; l1++) {
                            for (int l2 = 0; l2 < nn2; l2++) {
                                int node_i = graph[num_delta_spaces*best_indices[0]+best_indices[1]][i][k][l1];
                                int node_j = graph[num_delta_spaces*best_indices[0]+best_indices[1]][j][k][l2];
                                if (node_i % K_folds != test_fold and node_j % K_folds != test_fold) {
                                    weight += (data[1][node_i]-data[1][node_j])*(data[1][node_i]-data[1][node_j]);
                                    count++;
                                }
                            }
                        }
                    }
                    if (count >= 1) {
                        weight = (weight + 2*alpha[best_indices[2]])/count;
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


        for (int i = nodes[best_indices[1]] - 1; i >= 0; i--) {
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

        // cout << "Graph density: " << graph_edges/(max_graph_edges) << endl;

        maximum_spanning_tree(Adj, "TreeAdjacencyMatrix.csv");

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
        for (int i = test_fold; i < num_samples; i += K_folds) {

            int t = data[0][i]/(delta_time[best_indices[0]]*time_scale);
            int m = data[7][i]/delta_space[best_indices[1]];
            int n = data[6][i]/delta_space[best_indices[1]];
            double interpolated_value = 0;
            double total_weight = 0;
            int node = correspondence[N[best_indices[1]]*m+n];
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
                                if(graph_copy[num_delta_spaces*best_indices[0]+best_indices[1]][correspondence_back[j]][t].size() != 0) {
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
                    for (int k = 0; k < graph_copy[num_delta_spaces*best_indices[0]+best_indices[1]][correspondence_back[last_node]][t].size(); k++) {
                        mean += data[1][graph_copy[num_delta_spaces*best_indices[0]+best_indices[1]][correspondence_back[last_node]][t][k]];
                    }
                    mean /= graph_copy[num_delta_spaces*best_indices[0]+best_indices[1]][correspondence_back[last_node]][t].size();
                    interpolated_value += mean / combined_weight;
                }
                if (total_weight != 0) {
                    interpolated_value /= total_weight;
                    gsp_test_error += (interpolated_value - data[1][i])*(interpolated_value - data[1][i]);
                    gsp_test_count++;
                    interpolated_samples.push_back(i);
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

    for (int i = 0; i < K_folds; i++) {
        cout << test_results[i] << " " << (i == K_folds-1 ? "\n" : "");
    }
    cout << "FINAL RESULTS:\nMean test RMSE = " << mean_test_rmse << endl;
    
    auto end_time = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);

    cout << "Execution time: " << duration.count() <<  " seconds" << endl;

}
