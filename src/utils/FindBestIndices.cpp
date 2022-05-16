#include "utils.h"

vector<int> find_best_indices(const vector<vector<vector<vector<double>>>>& v) {
    vector<int> result {-1, -1, -1, -1};
    int num_delta_times = v.size();
    int num_delta_spaces = v[0].size();
    int num_alphas = v[0][0].size();
    int num_max_dist = v[0][0][0].size();
    cout << num_delta_times << " " << num_delta_spaces << " " << num_alphas << endl;
    double min_value = INT_MAX;
    for (int i = 0; i < num_delta_times; i++) {
        for (int j = 0; j < num_delta_spaces; j++) {
            for (int a = 0; a < num_alphas; a++) {
                for (int d = 0; d < num_max_dist; d++) {
                    if (v[i][j][a][d] < min_value) {
                        min_value = v[i][j][a][d];
                        result = {i, j, a, d};
                    }
                }
            }
        }
    }
    return result;
}