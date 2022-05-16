#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <math.h>
#include <map>
#include <chrono>
using namespace std;

/**
 * @brief Reads a .csv file containing numeric data.
 * 
 * @param filename name of the .csv file.
 * @return vector<vector<double>> matrix containing the data in the .csv file.
 *         Each row represents a measured variable (time, temperature, etc.).
 *         Each column represents a set of values corresponding to an observation.
 */
vector<vector<double>> read_CSV(string filename);

/**
 * @brief Computes the Haversine distance in meters between two coordinates.
 * 
 * @param lat1 latitude of point 1.
 * @param lon1 longitude of point 1.
 * @param lat2 latitude of point 2.
 * @param lon2 longitude of point 2.
 * @return double value in meters of the Haversine distance between both coordinates.
 */
double haversine_distance(double lat1, double lon1, double lat2, double lon2);

/**
 * @brief Auxiliar function to find the index of max-weight vertex from set of unvisited vertices.
 * 
 * @param visited vector of boolean with visited vertices marked as true and unvisited as false.
 * @param weights vector of doubles containing weight values.
 * @param N number of vertices in the graph.
 * @return int index of the vertex with the maximum weight.
 */
int find_max_vertex(vector<bool>& visited, vector<double>& weights, int N);

/**
 * @brief Utility function to save the maximum spanning tree in a .csv file.
 * 
 * @param graph adjacency matrix of the graph containing the maximum spanning tree.
 * @param parent vector of parent connections between nodes.
 * @param N number of vertices in the graph.
 * @param filename name of the saved file containing the maximum spanning tree.
 */
void print_maximum_spanning_tree(const vector<vector<double>>& graph, const vector<int>& parent, int N, string filename);

/**
 * @brief Main function to calculate the maximum spanning tree and save it as a .csv file.
 * 
 * @param graph adjacency matrix containing all the computed weights between each pair of nodes.
 * @param filename name of the saved file containing the maximum spanning tree.
 */
void maximum_spanning_tree(const vector<vector<double>>& graph, string filename);

/**
 * @brief Returns the best indices for each model hyper-parameter given a 4-Dimensional matrix 
 *        containing the RMSE obtained for each combination of hyper parameters.
 * 
 * @param v 4-D matrix containing the RMSE values for each combination of hyper-parameters.
 * @return vector<int> vector with 4 integers, each one referring to the index providing lowest RMSE
 *         for the corresponding hyper-parameter dimension.
 */
vector<int> find_best_indices(const vector<vector<vector<vector<double>>>>& v);