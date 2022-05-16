#include "utils.h"

double haversine_distance(double lat1, double lon1, double lat2, double lon2) {
    double R = 6.3781e6;
    double delta_lat = abs(lat1-lat2);
    double delta_lon = abs(lon1-lon2);
    double a = sin(delta_lat/2) * sin(delta_lat/2) + cos(lat1) * cos(lat2) * sin(delta_lon/2) * sin(delta_lon/2);
    double c = 2 * atan2(sqrt(a),sqrt(1-a));
    return R * c;
}