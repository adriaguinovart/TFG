#include "utils/utils.h"

int main() {

    cout << "INFO - Execution started" << endl;
    // Variable to quantify the execution time
    auto start_time = chrono::high_resolution_clock::now();
    
    int K_folds = 10;

    int delta_time = 60 * 120;
    int delta_space = 150;
    double alpha = 0.1;
    double max_edge_distance = 1000;

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
    int max_time = data[0][num_samples-1]; // Highest timestamp in the dataset

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

        if (t < T and m < M and n < N) {
            graph[N*m+n][t].push_back(i);
            node_samples[N*m+n]++;
        }
    }

    // Inverse Distance Weighting (IDW) interpolation
    double idw_error = 0;
    int idw_count = 0;
    vector<double> se;
    for (int i = 0; i < num_samples; i += K_folds) {

        int t = data[0][i]/(delta_time*time_scale);
        if (t < 4*T_week) {
            double interpolated_value = 0;
            double total_weight = 0;

            for (int j = 0; j < M*N; j++) {
                for (int k = 0; k < graph[j][t].size(); k++) {

                    int l = graph[j][t][k];

                    if (l % K_folds != 0) {

                        double dist = haversine_distance(M_PI / 180 * data[4][i], M_PI / 180 * data[5][i],
                                                M_PI / 180 * data[4][l], M_PI / 180 * data[5][l]);
                        if (dist == 0) {
                            dist = 1e-3; // Set distance to small value to avoid dividing by 0
                        }

                        interpolated_value += data[1][l]*(1/dist);
                        total_weight += 1/dist;
                    }
                }
            }
            interpolated_value /= total_weight;
            idw_error += (interpolated_value - data[1][i])*(interpolated_value - data[1][i]);
            se.push_back((interpolated_value - data[1][i])*(interpolated_value - data[1][i]));
            idw_count++;
        }
    }
    idw_error /= idw_count;
    double idw_rmse = sqrt(idw_error);
    cout << "IDW --> RMSE = " << idw_rmse << endl;

    double se_mean = 0;
    for (int i = 0; i < se.size(); i++){
        se_mean += se[i];
    }
    
    se_mean /= se.size();
    double mse_std = 0;
    for (int i = 0; i < se.size(); i++)
        mse_std += (se[i]-se_mean)*(se[i]-se_mean);
    mse_std /= se.size();

    cout << "Std: " << sqrt(mse_std) << endl;

    auto end_time = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "Execution time: " << double(duration.count())/1000 <<  " seconds" << endl;
    
}