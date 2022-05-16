#include "utils.h"

vector<vector<double>> read_CSV(string filename) {
    ifstream myFile(filename);

    if(!myFile.is_open()) cout << "Could not open file" << endl;

    vector<vector<double>> result;
    string line, colname;
    double val;

    if(myFile.good()) {
        getline(myFile, line);
        stringstream ss(line);
        while(getline(ss, colname, ',')){
            result.push_back(vector<double> {});
        }
    }
    while(getline(myFile, line)) {
            stringstream ss(line);
            int colIdx = 0;
            while(ss >> val){
                result.at(colIdx).push_back(val);
                if(ss.peek() == ',') ss.ignore();
                colIdx++;
            }
    }
    myFile.close();
    return result;
}
