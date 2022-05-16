#include "utils.h"


// Function to find index of max-weight vertex from set of unvisited vertices
int find_max_vertex(vector<bool>& visited, vector<double>& weights, int N) {
    // Stores the index of max-weight vertex from set of unvisited vertices
    int index = -1;
 
    // Stores the maximum weight from the set of unvisited vertices
    double maxW = INT_MIN;
    // Iterate over all possible nodes of a graph
    for (int i = 0; i < N; i++) {
        
        // If the current node is unvisited and weight of current vertex is greater than maxW
        if (visited[i] == false and weights[i] > maxW) {
 
            // Update maxW
            maxW = weights[i];
 
            // Update index
            index = i;
        }
    }
    return index;
}


// Utility function to find the maximum spanning tree of graph
void print_maximum_spanning_tree(const vector<vector<double>>& graph, const vector<int>& parent, int N, string filename)
{
 
    // Total weight of Maximum Spanning Tree
    /*double MST = 0;
    for (int i = 1; i < N; i++)
        MST += graph[i][parent[i]];
 
    cout << "Weight of the Maximum Spanning Tree " << MST << endl;
    cout << "Average edge weight " << MST/(N-1) << endl;*/
    
    ofstream my_file(filename);
    my_file << "node1,node2,weight" << endl;
    for (int i = 1; i < N; i++) {
        my_file << parent[i]+1 << ',' << i+1 << ',' << graph[i][parent[i]] << endl;
    }
    my_file.close();
}

void maximum_spanning_tree(const vector<vector<double>>& graph, string filename) {
    
    double N = graph.size();    
    
    // visited[i]: Check if vertex i is visited or not
    vector<bool> visited(N, false);

    // weights[i]: Stores maximum weight of graph to connect an edge with i
    vector<double> weights(N,INT_MIN);
 
    // parent[i]: Stores the parent node of vertex i
    vector<int> parent(N);
 
    // Include 1st vertex in maximum spanning tree
    weights[0] = INT_MAX;
    parent[0] = -1;

    // Search for other (V-1) vertices and build a tree
    for (int i = 0; i < N - 1; i++) {
        // Stores index of max-weight vertex from a set of unvisited vertex
        int max_vertex_index = find_max_vertex(visited, weights, N);
        if (max_vertex_index == -1) {
            break;
        }
        // Mark that vertex as visited
        visited[max_vertex_index] = true;
 
        // Update adjacent vertices of the current visited vertex
        for (int j = 0; j < N; j++) {
            // If there is an edge between j and current visited vertex and also j is unvisited vertex
            if (graph[j][max_vertex_index] != 0 and visited[j] == false) {
 
                // If graph[j][max_vertex_index] is greater than weights[j]
                if (graph[j][max_vertex_index] > weights[j]) {
 
                    // Update weights[j]
                    weights[j] = graph[j][max_vertex_index];
 
                    // Update parent[j]
                    parent[j] = max_vertex_index;
                }
            }
        }
    }

    // Print maximum spanning tree
    print_maximum_spanning_tree(graph, parent, N, filename);
}
