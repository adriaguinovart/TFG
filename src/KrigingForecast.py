import pykrige.ok as pyok
import numpy as np
import pandas as pd
from math import floor, sqrt
import time

import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist
import warnings

###### Variogram models ######


def linear_variogram_model(m, d):
    """Linear model, m is [slope, nugget]"""
    slope = float(m[0])
    nugget = float(m[1])
    return slope * d + nugget


def power_variogram_model(m, d):
    """Power model, m is [scale, exponent, nugget]"""
    scale = float(m[0])
    exponent = float(m[1])
    nugget = float(m[2])
    return scale * d ** exponent + nugget


def gaussian_variogram_model(m, d):
    """Gaussian model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1.0 - np.exp(-(d ** 2.0) / (range_ * 4.0 / 7.0) ** 2.0)) + nugget


def exponential_variogram_model(m, d):
    """Exponential model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return psill * (1.0 - np.exp(-d / (range_ / 3.0))) + nugget


def spherical_variogram_model(m, d):
    """Spherical model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return np.piecewise(
        d,
        [d <= range_, d > range_],
        [
            lambda x: psill
            * ((3.0 * x) / (2.0 * range_) - (x ** 3.0) / (2.0 * range_ ** 3.0))
            + nugget,
            psill + nugget,
        ],
    )


def hole_effect_variogram_model(m, d):
    """Hole Effect model, m is [psill, range, nugget]"""
    psill = float(m[0])
    range_ = float(m[1])
    nugget = float(m[2])
    return (
        psill * (1.0 - (1.0 - d / (range_ / 3.0)) * np.exp(-d / (range_ / 3.0)))
        + nugget
    )


###### Core ######

from scipy.spatial.distance import pdist, squareform, cdist
from scipy.optimize import least_squares
import scipy.linalg as spl


eps = 1.0e-10  # Cutoff for comparison to zero


P_INV = {"pinv": spl.pinv, "pinv2": spl.pinv2, "pinvh": spl.pinvh}


def great_circle_distance(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between one or multiple pairs of
    points given in spherical coordinates. Spherical coordinates are expected
    in degrees. Angle definition follows standard longitude/latitude definition.
    This uses the arctan version of the great-circle distance function
    (en.wikipedia.org/wiki/Great-circle_distance) for increased
    numerical stability.
    Parameters
    ----------
    lon1: float scalar or numpy array
        Longitude coordinate(s) of the first element(s) of the point
        pair(s), given in degrees.
    lat1: float scalar or numpy array
        Latitude coordinate(s) of the first element(s) of the point
        pair(s), given in degrees.
    lon2: float scalar or numpy array
        Longitude coordinate(s) of the second element(s) of the point
        pair(s), given in degrees.
    lat2: float scalar or numpy array
        Latitude coordinate(s) of the second element(s) of the point
        pair(s), given in degrees.
    Calculation of distances follows numpy elementwise semantics, so if
    an array of length N is passed, all input parameters need to be
    arrays of length N or scalars.
    Returns
    -------
    distance: float scalar or numpy array
        The great circle distance(s) (in degrees) between the
        given pair(s) of points.
    """
    # Convert to radians:
    lat1 = np.array(lat1) * np.pi / 180.0
    lat2 = np.array(lat2) * np.pi / 180.0
    dlon = (lon1 - lon2) * np.pi / 180.0

    # Evaluate trigonometric functions that need to be evaluated more
    # than once:
    c1 = np.cos(lat1)
    s1 = np.sin(lat1)
    c2 = np.cos(lat2)
    s2 = np.sin(lat2)
    cd = np.cos(dlon)

    # This uses the arctan version of the great-circle distance function
    # from en.wikipedia.org/wiki/Great-circle_distance for increased
    # numerical stability.
    # Formula can be obtained from [2] combining eqns. (14)-(16)
    # for spherical geometry (f=0).

    return (
        180.0
        / np.pi
        * np.arctan2(
            np.sqrt((c2 * np.sin(dlon)) ** 2 + (c1 * s2 - s1 * c2 * cd) ** 2),
            s1 * s2 + c1 * c2 * cd,
        )
    )


def euclid3_to_great_circle(euclid3_distance):
    """Convert euclidean distance between points on a unit sphere to
    the corresponding great circle distance.
    Parameters
    ----------
    euclid3_distance: float scalar or numpy array
        The euclidean three-space distance(s) between points on a
        unit sphere, thus between [0,2].
    Returns
    -------
    great_circle_dist: float scalar or numpy array
        The corresponding great circle distance(s) between the points.
    """
    # Eliminate some possible numerical errors:
    euclid3_distance[euclid3_distance > 2.0] = 2.0
    return 180.0 - 360.0 / np.pi * np.arccos(0.5 * euclid3_distance)


def _adjust_for_anisotropy(X, center, scaling, angle):
    """Adjusts data coordinates to take into account anisotropy.
    Can also be used to take into account data scaling. Angles are CCW about
    specified axes. Scaling is applied in rotated coordinate system.
    Parameters
    ----------
    X : ndarray
        float array [n_samples, n_dim], the input array of coordinates
    center : ndarray
        float array [n_dim], the coordinate of centers
    scaling : ndarray
        float array [n_dim - 1], the scaling of last two dimensions
    angle : ndarray
        float array [2*n_dim - 3], the anisotropy angle (degrees)
    Returns
    -------
    X_adj : ndarray
        float array [n_samples, n_dim], the X array adjusted for anisotropy.
    """

    center = np.asarray(center)[None, :]
    angle = np.asarray(angle) * np.pi / 180

    X -= center

    Ndim = X.shape[1]

    if Ndim == 1:
        raise NotImplementedError("Not implemnented yet?")
    elif Ndim == 2:
        stretch = np.array([[1, 0], [0, scaling[0]]])
        rot_tot = np.array(
            [
                [np.cos(-angle[0]), -np.sin(-angle[0])],
                [np.sin(-angle[0]), np.cos(-angle[0])],
            ]
        )
    elif Ndim == 3:
        stretch = np.array(
            [[1.0, 0.0, 0.0], [0.0, scaling[0], 0.0], [0.0, 0.0, scaling[1]]]
        )
        rotate_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(-angle[0]), -np.sin(-angle[0])],
                [0.0, np.sin(-angle[0]), np.cos(-angle[0])],
            ]
        )
        rotate_y = np.array(
            [
                [np.cos(-angle[1]), 0.0, np.sin(-angle[1])],
                [0.0, 1.0, 0.0],
                [-np.sin(-angle[1]), 0.0, np.cos(-angle[1])],
            ]
        )
        rotate_z = np.array(
            [
                [np.cos(-angle[2]), -np.sin(-angle[2]), 0.0],
                [np.sin(-angle[2]), np.cos(-angle[2]), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rot_tot = np.dot(rotate_z, np.dot(rotate_y, rotate_x))
    else:
        raise ValueError(
            "Adjust for anisotropy function doesn't support ND spaces where N>3"
        )
    X_adj = np.dot(stretch, np.dot(rot_tot, X.T)).T

    X_adj += center

    return X_adj


def _make_variogram_parameter_list(variogram_model, variogram_model_parameters):
    """Converts the user input for the variogram model parameters into the
    format expected in the rest of the code.
    Makes a list of variogram model parameters in the expected order if the
    user has provided the model parameters. If not, returns None, which
    will ensure that the automatic variogram estimation routine is
    triggered.
    Parameters
    ----------
    variogram_model : str
        specifies the variogram model type
    variogram_model_parameters : list, dict, or None
        parameters provided by the user, can also be None if the user
        did not specify the variogram model parameters; if None,
        this function returns None, that way the automatic variogram
        estimation routine will kick in down the road...
    Returns
    -------
    parameter_list : list
        variogram model parameters stored in a list in the expected order;
        if variogram_model is 'custom', model parameters should already
        be encapsulated in a list, so the list is returned unaltered;
        if variogram_model_parameters was not specified by the user,
        None is returned; order for internal variogram models is as follows...
        linear - [slope, nugget]
        power - [scale, exponent, nugget]
        gaussian - [psill, range, nugget]
        spherical - [psill, range, nugget]
        exponential - [psill, range, nugget]
        hole-effect - [psill, range, nugget]
    """

    if variogram_model_parameters is None:

        parameter_list = None

    elif type(variogram_model_parameters) is dict:

        if variogram_model in ["linear"]:

            if (
                "slope" not in variogram_model_parameters.keys()
                or "nugget" not in variogram_model_parameters.keys()
            ):

                raise KeyError(
                    "'linear' variogram model requires 'slope' "
                    "and 'nugget' specified in variogram model "
                    "parameter dictionary."
                )

            else:

                parameter_list = [
                    variogram_model_parameters["slope"],
                    variogram_model_parameters["nugget"],
                ]

        elif variogram_model in ["power"]:

            if (
                "scale" not in variogram_model_parameters.keys()
                or "exponent" not in variogram_model_parameters.keys()
                or "nugget" not in variogram_model_parameters.keys()
            ):

                raise KeyError(
                    "'power' variogram model requires 'scale', "
                    "'exponent', and 'nugget' specified in "
                    "variogram model parameter dictionary."
                )

            else:

                parameter_list = [
                    variogram_model_parameters["scale"],
                    variogram_model_parameters["exponent"],
                    variogram_model_parameters["nugget"],
                ]

        elif variogram_model in ["gaussian", "spherical", "exponential", "hole-effect"]:

            if (
                "range" not in variogram_model_parameters.keys()
                or "nugget" not in variogram_model_parameters.keys()
            ):

                raise KeyError(
                    "'%s' variogram model requires 'range', "
                    "'nugget', and either 'sill' or 'psill' "
                    "specified in variogram model parameter "
                    "dictionary." % variogram_model
                )

            else:

                if "sill" in variogram_model_parameters.keys():

                    parameter_list = [
                        variogram_model_parameters["sill"]
                        - variogram_model_parameters["nugget"],
                        variogram_model_parameters["range"],
                        variogram_model_parameters["nugget"],
                    ]

                elif "psill" in variogram_model_parameters.keys():

                    parameter_list = [
                        variogram_model_parameters["psill"],
                        variogram_model_parameters["range"],
                        variogram_model_parameters["nugget"],
                    ]

                else:

                    raise KeyError(
                        "'%s' variogram model requires either "
                        "'sill' or 'psill' specified in "
                        "variogram model parameter "
                        "dictionary." % variogram_model
                    )

        elif variogram_model in ["custom"]:

            raise TypeError(
                "For user-specified custom variogram model, "
                "parameters must be specified in a list, "
                "not a dict."
            )

        else:

            raise ValueError(
                "Specified variogram model must be one of the "
                "following: 'linear', 'power', 'gaussian', "
                "'spherical', 'exponential', 'hole-effect', "
                "'custom'."
            )

    elif type(variogram_model_parameters) is list:

        if variogram_model in ["linear"]:

            if len(variogram_model_parameters) != 2:

                raise ValueError(
                    "Variogram model parameter list must have "
                    "exactly two entries when variogram model "
                    "set to 'linear'."
                )

            parameter_list = variogram_model_parameters

        elif variogram_model in ["power"]:

            if len(variogram_model_parameters) != 3:

                raise ValueError(
                    "Variogram model parameter list must have "
                    "exactly three entries when variogram model "
                    "set to 'power'."
                )

            parameter_list = variogram_model_parameters

        elif variogram_model in ["gaussian", "spherical", "exponential", "hole-effect"]:

            if len(variogram_model_parameters) != 3:

                raise ValueError(
                    "Variogram model parameter list must have "
                    "exactly three entries when variogram model "
                    "set to '%s'." % variogram_model
                )

            parameter_list = [
                variogram_model_parameters[0] - variogram_model_parameters[2],
                variogram_model_parameters[1],
                variogram_model_parameters[2],
            ]

        elif variogram_model in ["custom"]:

            parameter_list = variogram_model_parameters

        else:

            raise ValueError(
                "Specified variogram model must be one of the "
                "following: 'linear', 'power', 'gaussian', "
                "'spherical', 'exponential', 'hole-effect', "
                "'custom'."
            )

    else:

        raise TypeError(
            "Variogram model parameters must be provided in either "
            "a list or a dict when they are explicitly specified."
        )

    return parameter_list


def _initialize_variogram_model(
    t,
    t2,
    X,
    y,
    variogram_model,
    variogram_model_parameters,
    variogram_function,
    nlags,
    weight,
    coordinates_type,
):
    """Initializes the variogram model for kriging. If user does not specify
    parameters, calls automatic variogram estimation routine.
    Returns lags, semivariance, and variogram model parameters.
    Parameters
    ----------
    X: ndarray
        float array [n_samples, n_dim], the input array of coordinates
    y: ndarray
        float array [n_samples], the input array of values to be kriged
    variogram_model: str
        user-specified variogram model to use
    variogram_model_parameters: list
        user-specified parameters for variogram model
    variogram_function: callable
        function that will be called to evaluate variogram model
        (only used if user does not specify variogram model parameters)
    nlags: int
        integer scalar, number of bins into which to group inter-point distances
    weight: bool
        boolean flag that indicates whether the semivariances at smaller lags
        should be weighted more heavily in the automatic variogram estimation
    coordinates_type: str
        type of coordinates in X array, can be 'euclidean' for standard
        rectangular coordinates or 'geographic' if the coordinates are lat/lon
    Returns
    -------
    lags: ndarray
        float array [nlags], distance values for bins into which the
        semivariances were grouped
    semivariance: ndarray
        float array [nlags], averaged semivariance for each bin
    variogram_model_parameters: list
        parameters for the variogram model, either returned unaffected if the
        user specified them or returned from the automatic variogram
        estimation routine
    """

    # distance calculation for rectangular coords now leverages
    # scipy.spatial.distance's pdist function, which gives pairwise distances
    # in a condensed distance vector (distance matrix flattened to a vector)
    # to calculate semivariances...
    if coordinates_type == "euclidean":
        cont = 0
        dists_array = []
        values_array = []
        for iterator in range(len(t)):
            dists_array.append(X[cont:(cont+t[iterator])])
            values_array.append(y[cont:(cont+t[iterator])])
            cont += t[iterator]
        cont = 0
        for iterator in range(len(t2)):
            dists_array.append(X[cont:(cont+t2[iterator])])
            values_array.append(y[cont:(cont+t2[iterator])])
            cont += t2[iterator]

       
        d = []
        g = []
        for i in range(len(dists_array)):
            d = np.concatenate((d, pdist(dists_array[i], metric="euclidean")))
            g = np.concatenate((g, 0.5 * pdist(values_array[i][:, None], metric="sqeuclidean")))

    # geographic coordinates only accepted if the problem is 2D
    # assume X[:, 0] ('x') => lon, X[:, 1] ('y') => lat
    # old method of distance calculation is retained here...
    # could be improved in the future
    elif coordinates_type == "geographic":
        if X.shape[1] != 2:
            raise ValueError(
                "Geographic coordinate type only supported for 2D datasets."
            )
        x1, x2 = np.meshgrid(X[:, 0], X[:, 0], sparse=True)
        y1, y2 = np.meshgrid(X[:, 1], X[:, 1], sparse=True)
        z1, z2 = np.meshgrid(y, y, sparse=True)
        d = great_circle_distance(x1, y1, x2, y2)
        g = 0.5 * (z1 - z2) ** 2.0
        indices = np.indices(d.shape)
        d = d[(indices[0, :, :] > indices[1, :, :])]
        g = g[(indices[0, :, :] > indices[1, :, :])]

    else:
        raise ValueError(
            "Specified coordinate type '%s' is not supported." % coordinates_type
        )

    # Equal-sized bins are now implemented. The upper limit on the bins
    # is appended to the list (instead of calculated as part of the
    # list comprehension) to avoid any numerical oddities
    # (specifically, say, ending up as 0.99999999999999 instead of 1.0).
    # Appending dmax + 0.001 ensures that the largest distance value
    # is included in the semivariogram calculation.
    dmax = np.amax(d)
    dmin = np.amin(d)
    dd = (dmax - dmin) / nlags
    bins = [dmin + n * dd for n in range(nlags)]
    dmax += 0.001
    bins.append(dmax)

    # This old binning method was experimental and doesn't seem
    # to work too well. Bins were computed such that there are more
    # at shorter lags. This effectively weights smaller distances more
    # highly in determining the variogram. As Kitanidis points out,
    # the variogram fit to the data at smaller lag distances is more
    # important. However, the value at the largest lag probably ends up
    # being biased too high for the larger values and thereby throws off
    # automatic variogram calculation and confuses comparison of the
    # semivariogram with the variogram model.
    #
    # dmax = np.amax(d)
    # dmin = np.amin(d)
    # dd = dmax - dmin
    # bins = [dd*(0.5**n) + dmin for n in range(nlags, 1, -1)]
    # bins.insert(0, dmin)
    # bins.append(dmax)

    lags = np.zeros(nlags)
    semivariance = np.zeros(nlags)

    for n in range(nlags):
        # This 'if... else...' statement ensures that there are data
        # in the bin so that numpy can actually find the mean. If we
        # don't test this first, then Python kicks out an annoying warning
        # message when there is an empty bin and we try to calculate the mean.
        if d[(d >= bins[n]) & (d < bins[n + 1])].size > 0:
            lags[n] = np.mean(d[(d >= bins[n]) & (d < bins[n + 1])])
            semivariance[n] = np.mean(g[(d >= bins[n]) & (d < bins[n + 1])])
        else:
            lags[n] = np.nan
            semivariance[n] = np.nan

    lags = lags[~np.isnan(semivariance)]
    semivariance = semivariance[~np.isnan(semivariance)]

    # a few tests the make sure that, if the variogram_model_parameters
    # are supplied, they have been supplied as expected...
    # if variogram_model_parameters was not defined, then estimate the variogram
    if variogram_model_parameters is not None:
        if variogram_model == "linear" and len(variogram_model_parameters) != 2:
            raise ValueError(
                "Exactly two parameters required for linear variogram model."
            )
        elif (
            variogram_model
            in ["power", "spherical", "exponential", "gaussian", "hole-effect"]
            and len(variogram_model_parameters) != 3
        ):
            raise ValueError(
                "Exactly three parameters required for "
                "%s variogram model" % variogram_model
            )
    else:
        if variogram_model == "custom":
            raise ValueError(
                "Variogram parameters must be specified when "
                "implementing custom variogram model."
            )
        else:
            variogram_model_parameters = _calculate_variogram_model(
                lags, semivariance, variogram_model, variogram_function, weight
            )

    return lags, semivariance, variogram_model_parameters


def _variogram_residuals(params, x, y, variogram_function, weight):
    """Function used in variogram model estimation. Returns residuals between
    calculated variogram and actual data (lags/semivariance).
    Called by _calculate_variogram_model.
    Parameters
    ----------
    params: list or 1D array
        parameters for calculating the model variogram
    x: ndarray
        lags (distances) at which to evaluate the model variogram
    y: ndarray
        experimental semivariances at the specified lags
    variogram_function: callable
        the actual funtion that evaluates the model variogram
    weight: bool
        flag for implementing the crude weighting routine, used in order to
        fit smaller lags better
    Returns
    -------
    resid: 1d array
        residuals, dimension same as y
    """

    # this crude weighting routine can be used to better fit the model
    # variogram to the experimental variogram at smaller lags...
    # the weights are calculated from a logistic function, so weights at small
    # lags are ~1 and weights at the longest lags are ~0;
    # the center of the logistic weighting is hard-coded to be at 70% of the
    # distance from the shortest lag to the largest lag
    if weight:
        drange = np.amax(x) - np.amin(x)
        k = 2.1972 / (0.1 * drange)
        x0 = 0.7 * drange + np.amin(x)
        weights = 1.0 / (1.0 + np.exp(-k * (x0 - x)))
        weights /= np.sum(weights)
        resid = (variogram_function(params, x) - y) * weights
    else:
        resid = variogram_function(params, x) - y

    return resid


def _calculate_variogram_model(
    lags, semivariance, variogram_model, variogram_function, weight
):
    """Function that fits a variogram model when parameters are not specified.
    Returns variogram model parameters that minimize the RMSE between the
    specified variogram function and the actual calculated variogram points.
    Parameters
    ----------
    lags: 1d array
        binned lags/distances to use for variogram model parameter estimation
    semivariance: 1d array
        binned/averaged experimental semivariances to use for variogram model
        parameter estimation
    variogram_model: str/unicode
        specified variogram model to use for parameter estimation
    variogram_function: callable
        the actual funtion that evaluates the model variogram
    weight: bool
        flag for implementing the crude weighting routine, used in order to fit
        smaller lags better this is passed on to the residual calculation
        cfunction, where weighting is actually applied...
    Returns
    -------
    res: list
        list of estimated variogram model parameters
    NOTE that the estimation routine works in terms of the partial sill
    (psill = sill - nugget) -- setting bounds such that psill > 0 ensures that
    the sill will always be greater than the nugget...
    """

    if variogram_model == "linear":
        x0 = [
            (np.amax(semivariance) - np.amin(semivariance))
            / (np.amax(lags) - np.amin(lags)),
            np.amin(semivariance),
        ]
        bnds = ([0.0, 0.0], [np.inf, np.amax(semivariance)])
    elif variogram_model == "power":
        x0 = [
            (np.amax(semivariance) - np.amin(semivariance))
            / (np.amax(lags) - np.amin(lags)),
            1.1,
            np.amin(semivariance),
        ]
        bnds = ([0.0, 0.001, 0.0], [np.inf, 1.999, np.amax(semivariance)])
    else:
        x0 = [
            np.amax(semivariance) - np.amin(semivariance),
            0.25 * np.amax(lags),
            np.amin(semivariance),
        ]
        bnds = (
            [0.0, 0.0, 0.0],
            [10.0 * np.amax(semivariance), np.amax(lags), np.amax(semivariance)],
        )

    # use 'soft' L1-norm minimization in order to buffer against
    # potential outliers (weird/skewed points)
    res = least_squares(
        _variogram_residuals,
        x0,
        bounds=bnds,
        loss="soft_l1",
        args=(lags, semivariance, variogram_function, weight),
    )

    return res.x


def _krige(
    X,
    y,
    coords,
    variogram_function,
    variogram_model_parameters,
    coordinates_type,
    pseudo_inv=False,
):
    """Sets up and solves the ordinary kriging system for the given
    coordinate pair. This function is only used for the statistics calculations.
    Parameters
    ----------
    X: ndarray
        float array [n_samples, n_dim], the input array of coordinates
    y: ndarray
        float array [n_samples], the input array of measurement values
    coords: ndarray
        float array [1, n_dim], point at which to evaluate the kriging system
    variogram_function: callable
        function that will be called to evaluate variogram model
    variogram_model_parameters: list
        user-specified parameters for variogram model
    coordinates_type: str
        type of coordinates in X array, can be 'euclidean' for standard
        rectangular coordinates or 'geographic' if the coordinates are lat/lon
    pseudo_inv : :class:`bool`, optional
        Whether the kriging system is solved with the pseudo inverted
        kriging matrix. If `True`, this leads to more numerical stability
        and redundant points are averaged. But it can take more time.
        Default: False
    Returns
    -------
    zinterp: float
        kriging estimate at the specified point
    sigmasq: float
        mean square error of the kriging estimate
    """

    zero_index = None
    zero_value = False

    # calculate distance between points... need a square distance matrix
    # of inter-measurement-point distances and a vector of distances between
    # measurement points (X) and the kriging point (coords)
    if coordinates_type == "euclidean":
        d = squareform(pdist(X, metric="euclidean"))
        bd = np.squeeze(cdist(X, coords[None, :], metric="euclidean"))

    # geographic coordinate distances still calculated in the old way...
    # assume X[:, 0] ('x') => lon, X[:, 1] ('y') => lat
    # also assume problem is 2D; check done earlier in initializing variogram
    elif coordinates_type == "geographic":
        x1, x2 = np.meshgrid(X[:, 0], X[:, 0], sparse=True)
        y1, y2 = np.meshgrid(X[:, 1], X[:, 1], sparse=True)
        d = great_circle_distance(x1, y1, x2, y2)
        bd = great_circle_distance(
            X[:, 0],
            X[:, 1],
            coords[0] * np.ones(X.shape[0]),
            coords[1] * np.ones(X.shape[0]),
        )

    # this check is done when initializing variogram, but kept here anyways...
    else:
        raise ValueError(
            "Specified coordinate type '%s' is not supported." % coordinates_type
        )

    # check if kriging point overlaps with measurement point
    if np.any(np.absolute(bd) <= 1e-10):
        zero_value = True
        zero_index = np.where(bd <= 1e-10)[0][0]

    # set up kriging matrix
    n = X.shape[0]
    a = np.zeros((n + 1, n + 1))
    a[:n, :n] = -variogram_function(variogram_model_parameters, d)
    np.fill_diagonal(a, 0.0)
    a[n, :] = 1.0
    a[:, n] = 1.0
    a[n, n] = 0.0

    # set up RHS
    b = np.zeros((n + 1, 1))
    b[:n, 0] = -variogram_function(variogram_model_parameters, bd)
    if zero_value:
        b[zero_index, 0] = 0.0
    b[n, 0] = 1.0

    # solve
    if pseudo_inv:
        res = np.linalg.lstsq(a, b, rcond=None)[0]
    else:
        res = np.linalg.solve(a, b)
    zinterp = np.sum(res[:n, 0] * y)
    sigmasq = np.sum(res[:, 0] * -b[:, 0])

    return zinterp, sigmasq


def _find_statistics(
    X,
    y,
    variogram_function,
    variogram_model_parameters,
    coordinates_type,
    pseudo_inv=False,
):
    """Calculates variogram fit statistics.
    Returns the delta, sigma, and epsilon values for the variogram fit.
    These arrays are used for statistics calculations.
    Parameters
    ----------
    X: ndarray
        float array [n_samples, n_dim], the input array of coordinates
    y: ndarray
        float array [n_samples], the input array of measurement values
    variogram_function: callable
        function that will be called to evaluate variogram model
    variogram_model_parameters: list
        user-specified parameters for variogram model
    coordinates_type: str
        type of coordinates in X array, can be 'euclidean' for standard
        rectangular coordinates or 'geographic' if the coordinates are lat/lon
    pseudo_inv : :class:`bool`, optional
        Whether the kriging system is solved with the pseudo inverted
        kriging matrix. If `True`, this leads to more numerical stability
        and redundant points are averaged. But it can take more time.
        Default: False
    Returns
    -------
    delta: ndarray
        residuals between observed values and kriged estimates for those values
    sigma: ndarray
        mean error in kriging estimates
    epsilon: ndarray
        residuals normalized by their mean error
    """

    delta = np.zeros(y.shape)
    sigma = np.zeros(y.shape)

    for i in range(y.shape[0]):

        # skip the first value in the kriging problem
        if i == 0:
            continue

        else:
            k, ss = _krige(
                X[:i, :],
                y[:i],
                X[i, :],
                variogram_function,
                variogram_model_parameters,
                coordinates_type,
                pseudo_inv,
            )

            # if the estimation error is zero, it's probably because
            # the evaluation point X[i, :] is really close to one of the
            # kriging system points in X[:i, :]...
            # in the case of zero estimation error, the results are not stored
            if np.absolute(ss) < eps:
                continue

            delta[i] = y[i] - k
            sigma[i] = np.sqrt(ss)

    # only use non-zero entries in these arrays... sigma is used to pull out
    # non-zero entries in both cases because it is guaranteed to be positive,
    # whereas delta can be either positive or negative
    delta = delta[sigma > eps]
    sigma = sigma[sigma > eps]
    epsilon = delta / sigma

    return delta, sigma, epsilon


def calcQ1(epsilon):
    """Returns the Q1 statistic for the variogram fit (see [1])."""
    return abs(np.sum(epsilon) / (epsilon.shape[0] - 1))


def calcQ2(epsilon):
    """Returns the Q2 statistic for the variogram fit (see [1])."""
    return np.sum(epsilon ** 2) / (epsilon.shape[0] - 1)


def calc_cR(Q2, sigma):
    """Returns the cR statistic for the variogram fit (see [1])."""
    return Q2 * np.exp(np.sum(np.log(sigma ** 2)) / sigma.shape[0])

###### Ordinary Kriging class ######

class OrdinaryKriging:
    r"""Convenience class for easy access to 2D Ordinary Kriging.
    Parameters
    ----------
    x : array_like
        X-coordinates of data points.
    y : array_like
        Y-coordinates of data points.
    z : array-like
        Values at data points.
    variogram_model : str or GSTools CovModel, optional
        Specifies which variogram model to use; may be one of the following:
        linear, power, gaussian, spherical, exponential, hole-effect.
        Default is linear variogram model. To utilize a custom variogram model,
        specify 'custom'; you must also provide variogram_parameters and
        variogram_function. Note that the hole-effect model is only technically
        correct for one-dimensional problems.
        You can also use a
        `GSTools <https://github.com/GeoStat-Framework/GSTools>`_ CovModel.
    variogram_parameters : list or dict, optional
        Parameters that define the specified variogram model. If not provided,
        parameters will be automatically calculated using a "soft" L1 norm
        minimization scheme. For variogram model parameters provided in a dict,
        the required dict keys vary according to the specified variogram
        model: ::
           # linear
               {'slope': slope, 'nugget': nugget}
           # power
               {'scale': scale, 'exponent': exponent, 'nugget': nugget}
           # gaussian, spherical, exponential and hole-effect:
               {'sill': s, 'range': r, 'nugget': n}
               # OR
               {'psill': p, 'range': r, 'nugget': n}
        Note that either the full sill or the partial sill
        (psill = sill - nugget) can be specified in the dict.
        For variogram model parameters provided in a list, the entries
        must be as follows: ::
           # linear
               [slope, nugget]
           # power
               [scale, exponent, nugget]
           # gaussian, spherical, exponential and hole-effect:
               [sill, range, nugget]
        Note that the full sill (NOT the partial sill) must be specified
        in the list format.
        For a custom variogram model, the parameters are required, as custom
        variogram models will not automatically be fit to the data.
        Furthermore, the parameters must be specified in list format, in the
        order in which they are used in the callable function (see
        variogram_function for more information). The code does not check
        that the provided list contains the appropriate number of parameters
        for the custom variogram model, so an incorrect parameter list in
        such a case will probably trigger an esoteric exception someplace
        deep in the code.
        NOTE that, while the list format expects the full sill, the code
        itself works internally with the partial sill.
    variogram_function : callable, optional
        A callable function that must be provided if variogram_model is
        specified as 'custom'. The function must take only two arguments:
        first, a list of parameters for the variogram model; second, the
        distances at which to calculate the variogram model. The list
        provided in variogram_parameters will be passed to the function
        as the first argument.
    nlags : int, optional
        Number of averaging bins for the semivariogram. Default is 6.
    weight : bool, optional
        Flag that specifies if semivariance at smaller lags should be weighted
        more heavily when automatically calculating variogram model.
        The routine is currently hard-coded such that the weights are
        calculated from a logistic function, so weights at small lags are ~1
        and weights at the longest lags are ~0; the center of the logistic
        weighting is hard-coded to be at 70% of the distance from the shortest
        lag to the largest lag. Setting this parameter to True indicates that
        weights will be applied. Default is False. (Kitanidis suggests that the
        values at smaller lags are more important in fitting a variogram model,
        so the option is provided to enable such weighting.)
    anisotropy_scaling : float, optional
        Scalar stretching value to take into account anisotropy.
        Default is 1 (effectively no stretching).
        Scaling is applied in the y-direction in the rotated data frame
        (i.e., after adjusting for the anisotropy_angle, if anisotropy_angle
        is not 0). This parameter has no effect if coordinate_types is
        set to 'geographic'.
    anisotropy_angle : float, optional
        CCW angle (in degrees) by which to rotate coordinate system in
        order to take into account anisotropy. Default is 0 (no rotation).
        Note that the coordinate system is rotated. This parameter has
        no effect if coordinate_types is set to 'geographic'.
    verbose : bool, optional
        Enables program text output to monitor kriging process.
        Default is False (off).
    enable_plotting : bool, optional
        Enables plotting to display variogram. Default is False (off).
    enable_statistics : bool, optional
        Default is False
    coordinates_type : str, optional
        One of 'euclidean' or 'geographic'. Determines if the x and y
        coordinates are interpreted as on a plane ('euclidean') or as
        coordinates on a sphere ('geographic'). In case of geographic
        coordinates, x is interpreted as longitude and y as latitude
        coordinates, both given in degree. Longitudes are expected in
        [0, 360] and latitudes in [-90, 90]. Default is 'euclidean'.
    exact_values : bool, optional
        If True, interpolation provides input values at input locations.
        If False, interpolation accounts for variance/nugget within input
        values at input locations and does not behave as an
        exact-interpolator [2]. Note that this only has an effect if
        there is variance/nugget present within the input data since it is
        interpreted as measurement error. If the nugget is zero, the kriged
        field will behave as an exact interpolator.
    pseudo_inv : :class:`bool`, optional
        Whether the kriging system is solved with the pseudo inverted
        kriging matrix. If `True`, this leads to more numerical stability
        and redundant points are averaged. But it can take more time.
        Default: False
    pseudo_inv_type : :class:`str`, optional
        Here you can select the algorithm to compute the pseudo-inverse matrix:
            * `"pinv"`: use `pinv` from `scipy` which uses `lstsq`
            * `"pinv2"`: use `pinv2` from `scipy` which uses `SVD`
            * `"pinvh"`: use `pinvh` from `scipy` which uses eigen-values
        Default: `"pinv"`
    References
    ----------
    .. [1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in
       Hydrogeology, (Cambridge University Press, 1997) 272 p.
    .. [2] N. Cressie, Statistics for spatial data,
       (Wiley Series in Probability and Statistics, 1993) 137 p.
    """

    eps = 1.0e-10  # Cutoff for comparison to zero
    variogram_dict = {
        "linear": linear_variogram_model,
        "power": power_variogram_model,
        "gaussian": gaussian_variogram_model,
        "spherical": spherical_variogram_model,
        "exponential": exponential_variogram_model,
        "hole-effect": hole_effect_variogram_model,
    }

    def __init__(
        self,
        t, t2,
        x,
        y,
        z,
        variogram_model="linear",
        variogram_parameters=None,
        variogram_function=None,
        nlags=6,
        weight=False,
        anisotropy_scaling=1.0,
        anisotropy_angle=0.0,
        verbose=False,
        enable_plotting=False,
        enable_statistics=False,
        coordinates_type="euclidean",
        exact_values=True,
        pseudo_inv=False,
        pseudo_inv_type="pinv",
    ):
        # config the pseudo inverse
        self.pseudo_inv = bool(pseudo_inv)
        self.pseudo_inv_type = str(pseudo_inv_type)
        if self.pseudo_inv_type not in P_INV:
            raise ValueError("pseudo inv type not valid: " + str(pseudo_inv_type))

        # set up variogram model and parameters...
        self.variogram_model = variogram_model
        self.model = None

        if not isinstance(exact_values, bool):
            raise ValueError("exact_values has to be boolean True or False")
        self.exact_values = exact_values

        # check if a GSTools covariance model is given
        if hasattr(self.variogram_model, "pykrige_kwargs"):
            # save the model in the class
            self.model = self.variogram_model
            if self.model.dim == 3:
                raise ValueError("GSTools: model dim is not 1 or 2")
            self.variogram_model = "custom"
            variogram_function = self.model.pykrige_vario
            variogram_parameters = []
            anisotropy_scaling = self.model.pykrige_anis
            anisotropy_angle = self.model.pykrige_angle
        if (
            self.variogram_model not in self.variogram_dict.keys()
            and self.variogram_model != "custom"
        ):
            raise ValueError(
                "Specified variogram model '%s' is not supported." % variogram_model
            )
        elif self.variogram_model == "custom":
            if variogram_function is None or not callable(variogram_function):
                raise ValueError(
                    "Must specify callable function for custom variogram model."
                )
            else:
                self.variogram_function = variogram_function
        else:
            self.variogram_function = self.variogram_dict[self.variogram_model]

        # Code assumes 1D input arrays of floats. Ensures that any extraneous
        # dimensions don't get in the way. Copies are created to avoid any
        # problems with referencing the original passed arguments.
        # Also, values are forced to be float... in the future, might be worth
        # developing complex-number kriging (useful for vector field kriging)
        self.X_ORIG = np.atleast_1d(
            np.squeeze(np.array(x, copy=True, dtype=np.float64))
        )
        self.Y_ORIG = np.atleast_1d(
            np.squeeze(np.array(y, copy=True, dtype=np.float64))
        )
        self.Z = np.atleast_1d(np.squeeze(np.array(z, copy=True, dtype=np.float64)))

        self.verbose = verbose
        self.enable_plotting = enable_plotting
        if self.enable_plotting and self.verbose:
            print("Plotting Enabled\n")

        # adjust for anisotropy... only implemented for euclidean (rectangular)
        # coordinates, as anisotropy is ambiguous for geographic coordinates...
        if coordinates_type == "euclidean":
            self.XCENTER = (np.amax(self.X_ORIG) + np.amin(self.X_ORIG)) / 2.0
            self.YCENTER = (np.amax(self.Y_ORIG) + np.amin(self.Y_ORIG)) / 2.0
            self.anisotropy_scaling = anisotropy_scaling
            self.anisotropy_angle = anisotropy_angle
            if self.verbose:
                print("Adjusting data for anisotropy...")
            self.X_ADJUSTED, self.Y_ADJUSTED = _adjust_for_anisotropy(
                np.vstack((self.X_ORIG, self.Y_ORIG)).T,
                [self.XCENTER, self.YCENTER],
                [self.anisotropy_scaling],
                [self.anisotropy_angle],
            ).T
        elif coordinates_type == "geographic":
            # Leave everything as is in geographic case.
            # May be open to discussion?
            if anisotropy_scaling != 1.0:
                warnings.warn(
                    "Anisotropy is not compatible with geographic "
                    "coordinates. Ignoring user set anisotropy.",
                    UserWarning,
                )
            self.XCENTER = 0.0
            self.YCENTER = 0.0
            self.anisotropy_scaling = 1.0
            self.anisotropy_angle = 0.0
            self.X_ADJUSTED = self.X_ORIG
            self.Y_ADJUSTED = self.Y_ORIG
        else:
            raise ValueError(
                "Only 'euclidean' and 'geographic' are valid "
                "values for coordinates-keyword."
            )
        self.coordinates_type = coordinates_type

        if self.verbose:
            print("Initializing variogram model...")

        vp_temp = _make_variogram_parameter_list(
            self.variogram_model, variogram_parameters
        )
        (
            self.lags,
            self.semivariance,
            self.variogram_model_parameters,
        ) = _initialize_variogram_model(
            t, t2,
            np.vstack((self.X_ADJUSTED, self.Y_ADJUSTED)).T,
            self.Z,
            self.variogram_model,
            vp_temp,
            self.variogram_function,
            nlags,
            weight,
            self.coordinates_type,
        )

        if self.verbose:
            print("Coordinates type: '%s'" % self.coordinates_type, "\n")
            if self.variogram_model == "linear":
                print("Using '%s' Variogram Model" % "linear")
                print("Slope:", self.variogram_model_parameters[0])
                print("Nugget:", self.variogram_model_parameters[1], "\n")
            elif self.variogram_model == "power":
                print("Using '%s' Variogram Model" % "power")
                print("Scale:", self.variogram_model_parameters[0])
                print("Exponent:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], "\n")
            elif self.variogram_model == "custom":
                print("Using Custom Variogram Model")
            else:
                print("Using '%s' Variogram Model" % self.variogram_model)
                print("Partial Sill:", self.variogram_model_parameters[0])
                print(
                    "Full Sill:",
                    self.variogram_model_parameters[0]
                    + self.variogram_model_parameters[2],
                )
                print("Range:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], "\n")
        if self.enable_plotting:
            self.display_variogram_model()

        if self.verbose:
            print("Calculating statistics on variogram model fit...")
        if enable_statistics:
            self.delta, self.sigma, self.epsilon = _find_statistics(
                np.vstack((self.X_ADJUSTED, self.Y_ADJUSTED)).T,
                self.Z,
                self.variogram_function,
                self.variogram_model_parameters,
                self.coordinates_type,
                self.pseudo_inv,
            )
            self.Q1 = calcQ1(self.epsilon)
            self.Q2 = calcQ2(self.epsilon)
            self.cR = calc_cR(self.Q2, self.sigma)
            if self.verbose:
                print("Q1 =", self.Q1)
                print("Q2 =", self.Q2)
                print("cR =", self.cR, "\n")
        else:
            self.delta, self.sigma, self.epsilon, self.Q1, self.Q2, self.cR = [None] * 6

    def update_variogram_model(
        self,
        variogram_model,
        variogram_parameters=None,
        variogram_function=None,
        nlags=6,
        weight=False,
        anisotropy_scaling=1.0,
        anisotropy_angle=0.0,
    ):
        """Allows user to update variogram type and/or
        variogram model parameters.
        Parameters
        __________
        variogram_model : str or GSTools CovModel
            May be any of the variogram models listed
            above. May also be 'custom', in which case variogram_parameters
            and variogram_function must be specified.
            You can also use a
            `GSTools <https://github.com/GeoStat-Framework/GSTools>`_ CovModel.
        variogram_parameters : list or dict, optional
            List or dict of
            variogram model parameters, as explained above. If not provided,
            a best fit model will be calculated as described above.
        variogram_function : callable, optional
            A callable function that must
            be provided if variogram_model is specified as 'custom'.
            See above for more information.
        nlags : int, optional
            Number of averaging bins for the semivariogram.
            Default is 6.
        weight : boolean, optional
            Flag that specifies if semivariance at
            smaller lags should be weighted more heavily when automatically
            calculating the variogram model. See above for more information.
            True indicates that weights will be applied. Default is False.
        anisotropy_scaling : float, optional
            Scalar stretching value to
            take into account anisotropy. Default is 1 (effectively no
            stretching). Scaling is applied in the y-direction.
        anisotropy_angle : float, optional
            CCW angle (in degrees) by
            which to rotate coordinate system in order to take into
            account anisotropy. Default is 0 (no rotation).
        """

        # set up variogram model and parameters...
        self.variogram_model = variogram_model
        self.model = None
        # check if a GSTools covariance model is given
        if hasattr(self.variogram_model, "pykrige_kwargs"):
            # save the model in the class
            self.model = self.variogram_model
            if self.model.dim == 3:
                raise ValueError("GSTools: model dim is not 1 or 2")
            self.variogram_model = "custom"
            variogram_function = self.model.pykrige_vario
            variogram_parameters = []
            anisotropy_scaling = self.model.pykrige_anis
            anisotropy_angle = self.model.pykrige_angle
        if (
            self.variogram_model not in self.variogram_dict.keys()
            and self.variogram_model != "custom"
        ):
            raise ValueError(
                "Specified variogram model '%s' is not supported." % variogram_model
            )
        elif self.variogram_model == "custom":
            if variogram_function is None or not callable(variogram_function):
                raise ValueError(
                    "Must specify callable function for custom variogram model."
                )
            else:
                self.variogram_function = variogram_function
        else:
            self.variogram_function = self.variogram_dict[self.variogram_model]

        if (
            anisotropy_scaling != self.anisotropy_scaling
            or anisotropy_angle != self.anisotropy_angle
        ):
            if self.coordinates_type == "euclidean":
                if self.verbose:
                    print("Adjusting data for anisotropy...")
                self.anisotropy_scaling = anisotropy_scaling
                self.anisotropy_angle = anisotropy_angle
                self.X_ADJUSTED, self.Y_ADJUSTED = _adjust_for_anisotropy(
                    np.vstack((self.X_ORIG, self.Y_ORIG)).T,
                    [self.XCENTER, self.YCENTER],
                    [self.anisotropy_scaling],
                    [self.anisotropy_angle],
                ).T
            elif self.coordinates_type == "geographic":
                if anisotropy_scaling != 1.0:
                    warnings.warn(
                        "Anisotropy is not compatible with geographic"
                        " coordinates. Ignoring user set anisotropy.",
                        UserWarning,
                    )
                self.anisotropy_scaling = 1.0
                self.anisotropy_angle = 0.0
                self.X_ADJUSTED = self.X_ORIG
                self.Y_ADJUSTED = self.Y_ORIG
        if self.verbose:
            print("Updating variogram mode...")

        # See note above about the 'use_psill' kwarg...
        vp_temp = _make_variogram_parameter_list(
            self.variogram_model, variogram_parameters
        )
        (
            self.lags,
            self.semivariance,
            self.variogram_model_parameters,
        ) = _initialize_variogram_model(
            t,
            np.vstack((self.X_ADJUSTED, self.Y_ADJUSTED)).T,
            self.Z,
            self.variogram_model,
            vp_temp,
            self.variogram_function,
            nlags,
            weight,
            self.coordinates_type,
        )

        if self.verbose:
            print("Coordinates type: '%s'" % self.coordinates_type, "\n")
            if self.variogram_model == "linear":
                print("Using '%s' Variogram Model" % "linear")
                print("Slope:", self.variogram_model_parameters[0])
                print("Nugget:", self.variogram_model_parameters[1], "\n")
            elif self.variogram_model == "power":
                print("Using '%s' Variogram Model" % "power")
                print("Scale:", self.variogram_model_parameters[0])
                print("Exponent:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], "\n")
            elif self.variogram_model == "custom":
                print("Using Custom Variogram Model")
            else:
                print("Using '%s' Variogram Model" % self.variogram_model)
                print("Partial Sill:", self.variogram_model_parameters[0])
                print(
                    "Full Sill:",
                    self.variogram_model_parameters[0]
                    + self.variogram_model_parameters[2],
                )
                print("Range:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], "\n")
        if self.enable_plotting:
            self.display_variogram_model()

        if self.verbose:
            print("Calculating statistics on variogram model fit...")
        self.delta, self.sigma, self.epsilon = _find_statistics(
            np.vstack((self.X_ADJUSTED, self.Y_ADJUSTED)).T,
            self.Z,
            self.variogram_function,
            self.variogram_model_parameters,
            self.coordinates_type,
            self.pseudo_inv,
        )
        self.Q1 = calcQ1(self.epsilon)
        self.Q2 = calcQ2(self.epsilon)
        self.cR = calc_cR(self.Q2, self.sigma)
        if self.verbose:
            print("Q1 =", self.Q1)
            print("Q2 =", self.Q2)
            print("cR =", self.cR, "\n")

    def display_variogram_model(self):
        """Displays variogram model with the actual binned data."""
        import matplotlib.pyplot as plt
        plt.rc('font', family='serif')
        plt.rc('xtick',labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.grid(linestyle='--', linewidth=0.5)
        
        plt.xlabel("Distance (m)")
        plt.ylabel("Semivariance (ppb²)")

        sill = self.variogram_model_parameters[0] + self.variogram_model_parameters[2]
        range = self.variogram_model_parameters[1]
        nugget = self.variogram_model_parameters[2]
        plt.axvline(0, color='black', linestyle='-')
        plt.axhline(y=0, color='black', linestyle='-')
        plt.axvline(range, color='cadetblue', linestyle='--')
        plt.text(range + 10,sill/2,'Range',rotation=0)
        plt.axhline(y=nugget, color='cadetblue', linestyle='--')
        plt.text(range/2,nugget+0.05,'Nugget',rotation=0)
        plt.axhline(y=sill, color='cadetblue', linestyle='--')
        plt.text(range/2,sill+0.05,'Sill',rotation=0)

        ax.plot(self.lags, self.semivariance, "r^", markersize=5)
        ax.plot(
            self.lags,
            self.variogram_function(self.variogram_model_parameters, self.lags),
            "b-",
        ) 
        plt.show()



    def get_variogram_points(self):
        """Returns both the lags and the variogram function evaluated at each
        of them.
        The evaluation of the variogram function and the lags are produced
        internally. This method is convenient when the user wants to access to
        the lags and the resulting variogram (according to the model provided)
        for further analysis.
        Returns
        -------
        (tuple) tuple containing:
            lags (array) - the lags at which the variogram was evaluated
            variogram (array) - the variogram function evaluated at the lags
        """
        return (
            self.lags,
            self.variogram_function(self.variogram_model_parameters, self.lags),
        )

    def switch_verbose(self):
        """Allows user to switch code talk-back on/off. Takes no arguments."""
        self.verbose = not self.verbose

    def switch_plotting(self):
        """Allows user to switch plot display on/off. Takes no arguments."""
        self.enable_plotting = not self.enable_plotting

    def get_epsilon_residuals(self):
        """Returns the epsilon residuals for the variogram fit."""
        return self.epsilon

    def plot_epsilon_residuals(self):
        """Plots the epsilon residuals for the variogram fit."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(self.epsilon.size), self.epsilon, c="k", marker="*")
        ax.axhline(y=0.0)
        plt.show()

    def get_statistics(self):
        """Returns the Q1, Q2, and cR statistics for the variogram fit
        (in that order). No arguments.
        """
        return self.Q1, self.Q2, self.cR

    def print_statistics(self):
        """Prints out the Q1, Q2, and cR statistics for the variogram fit.
        NOTE that ideally Q1 is close to zero, Q2 is close to 1,
        and cR is as small as possible.
        """
        print("Q1 =", self.Q1)
        print("Q2 =", self.Q2)
        print("cR =", self.cR)

    def _get_kriging_matrix(self, n):
        """Assembles the kriging matrix."""

        if self.coordinates_type == "euclidean":
            xy = np.concatenate(
                (self.X_ADJUSTED[:, np.newaxis], self.Y_ADJUSTED[:, np.newaxis]), axis=1
            )
            d = cdist(xy, xy, "euclidean")
        elif self.coordinates_type == "geographic":
            d = great_circle_distance(
                self.X_ADJUSTED[:, np.newaxis],
                self.Y_ADJUSTED[:, np.newaxis],
                self.X_ADJUSTED,
                self.Y_ADJUSTED,
            )
        a = np.zeros((n + 1, n + 1))
        a[:n, :n] = -self.variogram_function(self.variogram_model_parameters, d)

        np.fill_diagonal(a, 0.0)
        a[n, :] = 1.0
        a[:, n] = 1.0
        a[n, n] = 0.0
        return a

    def _exec_vector(self, a, bd, mask):
        """Solves the kriging system as a vectorized operation. This method
        can take a lot of memory for large grids and/or large datasets."""

        npt = bd.shape[0]
        n = self.X_ADJUSTED.shape[0]
        zero_index = None
        zero_value = False

        # use the desired method to invert the kriging matrix
        if self.pseudo_inv:
            a_inv = P_INV[self.pseudo_inv_type](a)
        else:
            a_inv = scipy.linalg.inv(a)

        if np.any(np.absolute(bd) <= self.eps):
            zero_value = True
            zero_index = np.where(np.absolute(bd) <= self.eps)

        b = np.zeros((npt, n + 1, 1))
        b[:, :n, 0] = -self.variogram_function(self.variogram_model_parameters, bd)
        if zero_value and self.exact_values:
            b[zero_index[0], zero_index[1], 0] = 0.0
        b[:, n, 0] = 1.0

        if (~mask).any():
            mask_b = np.repeat(mask[:, np.newaxis, np.newaxis], n + 1, axis=1)
            b = np.ma.array(b, mask=mask_b)

        x = np.dot(a_inv, b.reshape((npt, n + 1)).T).reshape((1, n + 1, npt)).T
        zvalues = np.sum(x[:, :n, 0] * self.Z, axis=1)
        sigmasq = np.sum(x[:, :, 0] * -b[:, :, 0], axis=1)

        return zvalues, sigmasq

    def _exec_loop(self, a, bd_all, mask):
        """Solves the kriging system by looping over all specified points.
        Less memory-intensive, but involves a Python-level loop."""

        npt = bd_all.shape[0]
        n = self.X_ADJUSTED.shape[0]
        zvalues = np.zeros(npt)
        sigmasq = np.zeros(npt)

        # use the desired method to invert the kriging matrix
        if self.pseudo_inv:
            a_inv = P_INV[self.pseudo_inv_type](a)
        else:
            a_inv = scipy.linalg.inv(a)

        for j in np.nonzero(~mask)[
            0
        ]:  # Note that this is the same thing as range(npt) if mask is not defined,
            bd = bd_all[j]  # otherwise it takes the non-masked elements.
            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            else:
                zero_index = None
                zero_value = False

            b = np.zeros((n + 1, 1))
            b[:n, 0] = -self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value and self.exact_values:
                b[zero_index[0], 0] = 0.0
            b[n, 0] = 1.0
            x = np.dot(a_inv, b)
            zvalues[j] = np.sum(x[:n, 0] * self.Z)
            sigmasq[j] = np.sum(x[:, 0] * -b[:, 0])

        return zvalues, sigmasq

    def _exec_loop_moving_window(self, a_all, bd_all, mask, bd_idx):
        """Solves the kriging system by looping over all specified points.
        Less memory-intensive, but involves a Python-level loop."""
        import scipy.linalg.lapack

        npt = bd_all.shape[0]
        n = bd_idx.shape[1]
        zvalues = np.zeros(npt)
        sigmasq = np.zeros(npt)

        for i in np.nonzero(~mask)[
            0
        ]:  # Note that this is the same thing as range(npt) if mask is not defined,
            b_selector = bd_idx[i]  # otherwise it takes the non-masked elements.
            bd = bd_all[i]

            a_selector = np.concatenate((b_selector, np.array([a_all.shape[0] - 1])))
            a = a_all[a_selector[:, None], a_selector]

            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            else:
                zero_index = None
                zero_value = False
            b = np.zeros((n + 1, 1))
            b[:n, 0] = -self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value and self.exact_values:
                b[zero_index[0], 0] = 0.0
            b[n, 0] = 1.0

            x = scipy.linalg.solve(a, b)

            zvalues[i] = x[:n, 0].dot(self.Z[b_selector])
            sigmasq[i] = -x[:, 0].dot(b[:, 0])

        return zvalues, sigmasq

    def execute(
        self,
        style,
        xpoints,
        ypoints,
        mask=None,
        backend="vectorized",
        n_closest_points=None,
    ):
        pass
        """Calculates a kriged grid and the associated variance.
        Parameters
        ----------
        style : str
            Specifies how to treat input kriging points. Specifying 'grid'
            treats xpoints and ypoints as two arrays of x and y coordinates
            that define a rectangular grid. Specifying 'points' treats
            xpoints and ypoints as two arrays that provide coordinate pairs
            at which to solve the kriging system. Specifying 'masked'
            treats xpoints and ypoints as two arrays of x and y coordinates
            that define a rectangular grid and uses mask to only evaluate
            specific points in the grid.
        xpoints : array_like, shape (N,) or (N, 1)
            If style is specific as 'grid' or 'masked',
            x-coordinates of MxN grid. If style is specified as 'points',
            x-coordinates of specific points at which to solve
            kriging system.
        ypoints : array_like, shape (M,) or (M, 1)
            If style is specified as 'grid' or 'masked',
            y-coordinates of MxN grid. If style is specified as 'points',
            y-coordinates of specific points at which to solve kriging
            system. Note that in this case, xpoints and ypoints must have
            the same dimensions (i.e., M = N).
        mask : bool, array_like, shape (M, N), optional
            Specifies the points in the rectangular grid defined
            by xpoints and ypoints that are to be excluded in the
            kriging calculations. Must be provided if style is specified
            as 'masked'. False indicates that the point should not be
            masked, so the kriging system will be solved at the point.
            True indicates that the point should be masked, so the kriging
            system should will not be solved at the point.
        backend : str, optional
            Specifies which approach to use in kriging.
            Specifying 'vectorized' will solve the entire kriging problem
            at once in a vectorized operation. This approach is faster but
            also can consume a significant amount of memory for large grids
            and/or large datasets. Specifying 'loop' will loop through each
            point at which the kriging system is to be solved.
            This approach is slower but also less memory-intensive.
            Specifying 'C' will utilize a loop in Cython.
            Default is 'vectorized'.
        n_closest_points : int, optional
            For kriging with a moving window, specifies the number of
            nearby points to use in the calculation. This can speed up the
            calculation for large datasets, but should be used
            with caution. As Kitanidis notes, kriging with a moving window
            can produce unexpected oddities if the variogram model
            is not carefully chosen.
        Returns
        -------
        zvalues : ndarray, shape (M, N) or (N, 1)
            Z-values of specified grid or at the specified set of points.
            If style was specified as 'masked', zvalues will
            be a numpy masked array.
        sigmasq : ndarray, shape (M, N) or (N, 1)
            Variance at specified grid points or at the specified
            set of points. If style was specified as 'masked', sigmasq
            will be a numpy masked array.
        """
        '''
        if self.verbose:
            print("Executing Ordinary Kriging...\n")

        if style != "grid" and style != "masked" and style != "points":
            raise ValueError("style argument must be 'grid', 'points', or 'masked'")

        if n_closest_points is not None and n_closest_points <= 1:
            # If this is not checked, nondescriptive errors emerge
            # later in the code.
            raise ValueError("n_closest_points has to be at least two!")

        xpts = np.atleast_1d(np.squeeze(np.array(xpoints, copy=True)))
        ypts = np.atleast_1d(np.squeeze(np.array(ypoints, copy=True)))
        n = self.X_ADJUSTED.shape[0]
        nx = xpts.size
        ny = ypts.size
        a = self._get_kriging_matrix(n)

        if style in ["grid", "masked"]:
            if style == "masked":
                if mask is None:
                    raise IOError(
                        "Must specify boolean masking array when style is 'masked'."
                    )
                if mask.shape[0] != ny or mask.shape[1] != nx:
                    if mask.shape[0] == nx and mask.shape[1] == ny:
                        mask = mask.T
                    else:
                        raise ValueError(
                            "Mask dimensions do not match specified grid dimensions."
                        )
                mask = mask.flatten()
            npt = ny * nx
            grid_x, grid_y = np.meshgrid(xpts, ypts)
            xpts = grid_x.flatten()
            ypts = grid_y.flatten()

        elif style == "points":
            if xpts.size != ypts.size:
                raise ValueError(
                    "xpoints and ypoints must have "
                    "same dimensions when treated as "
                    "listing discrete points."
                )
            npt = nx
        else:
            raise ValueError("style argument must be 'grid', 'points', or 'masked'")

        if self.coordinates_type == "euclidean":
            xpts, ypts = _adjust_for_anisotropy(
                np.vstack((xpts, ypts)).T,
                [self.XCENTER, self.YCENTER],
                [self.anisotropy_scaling],
                [self.anisotropy_angle],
            ).T
            # Prepare for cdist:
            xy_data = np.concatenate(
                (self.X_ADJUSTED[:, np.newaxis], self.Y_ADJUSTED[:, np.newaxis]), axis=1
            )
            xy_points = np.concatenate(
                (xpts[:, np.newaxis], ypts[:, np.newaxis]), axis=1
            )
        elif self.coordinates_type == "geographic":
            # In spherical coordinates, we do not correct for anisotropy.
            # Also, we don't use scipy.spatial.cdist, so we do not have to
            # format the input data accordingly.
            pass

        if style != "masked":
            mask = np.zeros(npt, dtype="bool")

        c_pars = None
        if backend == "C":
            try:
                from .lib.cok import _c_exec_loop, _c_exec_loop_moving_window
            except ImportError:
                print(
                    "Warning: failed to load Cython extensions.\n"
                    "   See https://github.com/GeoStat-Framework/PyKrige/issues/8 \n"
                    "   Falling back to a pure python backend..."
                )
                backend = "loop"
            except:
                raise RuntimeError("Unknown error in trying to load Cython extension.")

            c_pars = {
                key: getattr(self, key)
                for key in [
                    "Z",
                    "eps",
                    "variogram_model_parameters",
                    "variogram_function",
                    "exact_values",
                    "pseudo_inv",
                    "pseudo_inv_type",
                ]
            }

        if n_closest_points is not None:
            if self.coordinates_type == "geographic":
                # To make use of the KDTree, we have to convert the
                # spherical coordinates into three dimensional Euclidean
                # coordinates, since the standard KDTree cannot handle
                # the periodicity.
                # Do the conversion just for the step involving the KDTree:
                lon_d = self.X_ADJUSTED[:, np.newaxis] * np.pi / 180.0
                lat_d = self.Y_ADJUSTED[:, np.newaxis] * np.pi / 180.0
                xy_data = np.concatenate(
                    (
                        np.cos(lon_d) * np.cos(lat_d),
                        np.sin(lon_d) * np.cos(lat_d),
                        np.sin(lat_d),
                    ),
                    axis=1,
                )
                lon_p = xpts[:, np.newaxis] * np.pi / 180.0
                lat_p = ypts[:, np.newaxis] * np.pi / 180.0
                xy_points = np.concatenate(
                    (
                        np.cos(lon_p) * np.cos(lat_p),
                        np.sin(lon_p) * np.cos(lat_p),
                        np.sin(lat_p),
                    ),
                    axis=1,
                )

            from scipy.spatial import cKDTree

            tree = cKDTree(xy_data)
            bd, bd_idx = tree.query(xy_points, k=n_closest_points, eps=0.0)

            if self.coordinates_type == "geographic":
                # Between the nearest neighbours from Euclidean search,
                # calculate the great circle distance using the standard method:
                x_points = np.tile(xpts[:, np.newaxis], (1, n_closest_points))
                y_points = np.tile(ypts[:, np.newaxis], (1, n_closest_points))
                bd = great_circle_distance(
                    x_points, y_points, self.X_ADJUSTED[bd_idx], self.Y_ADJUSTED[bd_idx]
                )

            if backend == "loop":
                zvalues, sigmasq = self._exec_loop_moving_window(a, bd, mask, bd_idx)
            elif backend == "C":
                zvalues, sigmasq = _c_exec_loop_moving_window(
                    a,
                    bd,
                    mask.astype("int8"),
                    bd_idx.astype(int),
                    self.X_ADJUSTED.shape[0],
                    c_pars,
                )
            else:
                raise ValueError(
                    "Specified backend {} for a moving window "
                    "is not supported.".format(backend)
                )
        else:
            if self.coordinates_type == "euclidean":
                bd = cdist(xy_points, xy_data, "euclidean")
            elif self.coordinates_type == "geographic":
                bd = great_circle_distance(
                    xpts[:, np.newaxis],
                    ypts[:, np.newaxis],
                    self.X_ADJUSTED,
                    self.Y_ADJUSTED,
                )

            if backend == "vectorized":
                zvalues, sigmasq = self._exec_vector(a, bd, mask)
            elif backend == "loop":
                zvalues, sigmasq = self._exec_loop(a, bd, mask)
            elif backend == "C":
                zvalues, sigmasq = _c_exec_loop(
                    a, bd, mask.astype("int8"), self.X_ADJUSTED.shape[0], c_pars
                )
            else:
                raise ValueError(
                    "Specified backend {} is not supported for "
                    "2D ordinary kriging.".format(backend)
                )

        if style == "masked":
            zvalues = np.ma.array(zvalues, mask=mask)
            sigmasq = np.ma.array(sigmasq, mask=mask)

        if style in ["masked", "grid"]:
            zvalues = zvalues.reshape((ny, nx))
            sigmasq = sigmasq.reshape((ny, nx))

        return zvalues, sigmasq'''


if __name__ == "__main__":
  # Variable to quantify the execution time
  start_time = time.time()

  # Random seed
  np.random.seed(100)

  # Model parameters
  delta_time = 3600 / 2
  T_week = 3600 * 24 * 7
  K_folds = 10
  num_time_frames = 1
  min_observations = 10

  # Enable usage of pseudo-inverse if delta_time is small
  if (delta_time > 3600 * 8):
    pseudo = False
  else:
    pseudo = True

  # Time resolution
  time_resolution = "seconds"
  if (time_resolution == "milliseconds"):
    time_scale = 1000
  elif (time_resolution == "seconds"):
    time_scale = 1
  else:
    time_scale = 0
    print("Unrecognized time scale")

  path = 'data/'
  file = 'processedData.csv'

  # Read data
  data = pd.read_csv(
            path + file,
            delim_whitespace=False, header=0,
            names=["generation_time","ozone_ppb", "temperature", "humidity", "latitude", "longitude", "delta_lat", "delta_lon"])


  max_time = data['generation_time'][len(data)-1]
  T = floor(T_week/(time_scale*delta_time))

  # Kriging model training and interpolation
  time_frame = data.loc[(data['generation_time'] < T*delta_time*time_scale )]
  if(len(time_frame) > min_observations):
    training_longitudes = np.array(time_frame['delta_lon'])
    training_latitudes = np.array(time_frame['delta_lat']) 
    training_values = np.array(time_frame['ozone_ppb'])
    t = []
    t2 = []
    for i in range(T):
      tf = time_frame.loc[(time_frame['generation_time'] >= i*delta_time*time_scale) & (time_frame['generation_time'] < (i+1)*delta_time*time_scale)]
      tf2 = time_frame.loc[(time_frame['generation_time'] >= (i-0.5)*delta_time*time_scale) & (time_frame['generation_time'] < (i+0.5)*delta_time*time_scale)]
      t.append(len(tf))
      t2.append(len(tf2))
      
    # Define Kriging model from training data
    OK = OrdinaryKriging(
                        t, t2,
                        training_longitudes,
                        training_latitudes,
                        training_values,
                        variogram_model="spherical",
                        verbose=False,
                        enable_plotting=True,
                        coordinates_type="euclidean",
                        nlags=20,
                        pseudo_inv=pseudo
                        )
  else:
    OK = -1
  
  ok_rmse_array = []
  ok_squared_error = []
  for t in range(T):
    # Interpolation of samples
    time_frame = data.loc[(data['generation_time'] >= (t+T)*delta_time*time_scale) & (data['generation_time'] < (t+1+T)*delta_time*time_scale)]
    if (OK != -1 and len(time_frame) > min_observations):
      folds = np.array_split(time_frame.reindex(np.random.permutation(time_frame.index)), K_folds)
      test_aux_data = []
      for i in range(1,K_folds):
        test_aux_data.append(folds[i])

      test_aux_data = pd.concat(test_aux_data, ignore_index=True)

      test_aux_longitudes = np.array(test_aux_data['delta_lon'])
      test_aux_latitudes = np.array(test_aux_data['delta_lat']) 
      test_aux_values = np.array(test_aux_data['ozone_ppb'])

      test_longitudes = np.array(folds[0]['delta_lon'])
      test_latitudes = np.array(folds[0]['delta_lat']) 
      test_values = np.array(folds[0]['ozone_ppb'])
              
      # Define a Kriging model with the test data and pre-computed optimal parameters
      OK_2 = pyok.OrdinaryKriging(
                              test_aux_longitudes,
                              test_aux_latitudes,
                              test_aux_values,
                              variogram_model="spherical",# linear, gaussian, spherical, exponential
                              variogram_parameters = {'psill' : OK.variogram_model_parameters[0] , 
                                                      'nugget' : OK.variogram_model_parameters[2], 
                                                      'range' : OK.variogram_model_parameters[1] },
                              variogram_function = OK.variogram_function,
                              verbose=False,
                              enable_plotting=False,
                              coordinates_type="euclidean",
                              nlags=20,
                              pseudo_inv=pseudo
                              )

      OK_2.lags = OK.lags
      OK_2.semivariance = OK.semivariance
      ok_predicted_values, ok_var = OK_2.execute("points", test_longitudes, test_latitudes)
      ok_squared_error = np.concatenate((ok_squared_error,(np.array(ok_predicted_values) - test_values) ** 2))
      
  
  mse = np.mean(ok_squared_error)
  rmse = sqrt(mse)
  print("Ordinary Kriging RMSE:", rmse, "MSE:", mse)
  var = 0
  for ei in ok_squared_error:
    var += (ei-np.mean(ok_squared_error))*(ei-np.mean(ok_squared_error))
  var /= len(ok_squared_error)

  std = np.std(ok_squared_error)
  
  # Display execution information
  print("Execution time:", "{:.2f}".format(time.time() - start_time), "seconds")
