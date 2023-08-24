import matplotlib
import numpy
import math
from orientationpy.plotting import projectOrientations3d

VERBOSE = False


def plotOrientations3d(
    orientations_zyx,
    projection="lambert",
    plot="both",
    binValueMin=None,
    binValueMax=None,
    binNormalisation=False,
    numberOfRings=9,
    pointMarkerSize=8,
    cmap=matplotlib.pyplot.cm.RdBu_r,
    title="",
    subtitle={"points": "", "bins": ""},
    saveFigPath=None,
    ax=None,  # Watch me

):
    """
    Main function for plotting 3D orientations.
    This function plots orientations (described by unit-direction vectors) from a sphere onto a plane.

    One useful trick for evaluating these orientations is to project them with a "Lambert equal area projection", which means that an isotropic distribution of angles is projected as equally filling the projected space.

    Parameters
    ----------
        orientations : Nx3 numpy array of floats
            Z, Y and X components of direction vectors.
            Non-unit vectors are normalised.

        projection : string, optional
            Selects different projection modes:
                **lambert** : Equal-area projection, default and highly reccommended. See https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection

                **equidistant** : equidistant projection

        plot : string, optional
            Selects which plots to show:
                **points** : shows projected points individually
                **bins** : shows binned orientations with counts inside each bin as colour
                **both** : shows both representations side-by-side, default

        title : string, optional
            Plot main title. Default = ""

        subtitle : dictionary, optional
            Sub-plot titles:
                **points** : Title for points plot. Default = ""
                **bins** : Title for bins plot. Default = ""

        binValueMin : int, optional
            Minimum colour-bar limits for bin view.
            Default = None (`i.e.`, auto-set)

        binValueMax : int, optional
            Maxmum colour-bar limits for bin view.
            Default = None (`i.e.`, auto-set)

        binNormalisation : bool, optional
            In binning mode, should bin counts be normalised by mean counts on all bins
            or absolute counts?

        cmap : matplotlib colour map, optional
            Colourmap for number of counts in each bin in the bin view.
            Default = ``matplotlib.pyplot.cm.RdBu_r``

        numberOfRings : int, optional
            Number of rings (`i.e.`, radial bins) for the bin view.
            The other bins are set automatically to have uniform sized bins using an algorithm from Jacquet and Tabot.
            Default = 9 (quite small bins)

        pointMarkerSize : int, optional
            Size of points in point view (5 OK for many points, 25 good for few points/debugging).
            Default = 8 (quite big points)

        saveFigPath : string, optional
            Path to save figure to -- stops the graphical plotting.
            Default = None

    Returns
    -------
        None -- A matplotlib graph is created and show()n

    Note
    ----
        Authors: Edward Andò, Hugues Talbot, Clara Jacquet and Max Wiebicke
    """
    import matplotlib.pyplot

    # ========================================================================
    # ==== Reading in data, and formatting to x,y,z sphere                 ===
    # ========================================================================
    numberOfPoints = orientations_zyx.shape[0]

    # ========================================================================
    # ==== Check that all the vectors are unit vectors                     ===
    # ========================================================================
    if VERBOSE:
        print("\t-> Normalising all vectors in x-y-z representation..."),

    # from http://stackoverflow.com/questions/2850743/numpy-how-to-quickly-normalize-many-vectors
    norms = numpy.apply_along_axis(numpy.linalg.norm, 1, orientations_zyx)
    print(norms.shape)
    print((norms.reshape(-1, 1)).shape)
    orientations_zyx = orientations_zyx / norms.reshape(-1, 1)

    print(orientations_zyx.shape)

    if VERBOSE:
        print("done.")

    # ========================================================================
    # ==== At this point we should have clean x,y,z data in memory         ===
    # ========================================================================
    if VERBOSE:
        print("\t-> We have %i orientations in memory." % (numberOfPoints))

    # Since this is the final number of vectors, at this point we can set up the
    #   matrices for the projection.
    projection_xy = numpy.zeros((numberOfPoints, 2))

    # TODO: Check if there are any values less than zero or more that 2*pi
    projection_theta_r = numpy.zeros((numberOfPoints, 2))

    # ========================================================================
    # ==== Projecting from x,y,z sphere to the desired projection          ===
    # ========================================================================
    # TODO: Vectorise this...
    for vectorN in range(numberOfPoints):
        # unpack 3D x,y,z
        z, y, x = orientations_zyx[vectorN]
        # print "\t\txyz = ", x, y, z

        # # fold over the negative half of the sphere
        # #     flip every component of the vector over
        # if z < 0:
        #     z = -z
        #     y = -y
        #     x = -x

        projection_xy[vectorN], projection_theta_r[vectorN] = projectOrientations3d([z, y, x], "cartesian", projection)
        
    # get radiusMax based on projection
    #                                    This is only limited to sqrt(2) because we're flipping over the negative side of the sphere
    if projection == "lambert":
        radiusMax = numpy.sqrt(2)
    elif projection == "stereo":
        radiusMax = 1.0
    elif projection == "equidistant":
        radiusMax = 1.0

    if VERBOSE:
        print("\t-> Biggest projected radius (r,t) = {}".format(numpy.abs(projection_theta_r[:, 1]).max()))

    # print "projection_xy\n", projection_xy
    # print "\n\nprojection_theta_r\n", projection_theta_r

    if plot == "points" or plot == "both":
        fig = matplotlib.pyplot.figure()
        fig.suptitle(title)

        ### --- Use existing axis --- ###
        show = ax is None # We assume if an axis wasn't passed, the plots should be shown.
        if ax is None:
            if plot == "both":
                ax = fig.add_subplot(121, polar=True)
            else:
                ax = fig.add_subplot(111, polar=True)

        ax.set_title(subtitle["points"] + "\n")

        # # set the line along which the numbers are plotted to 0°
        # # ax.set_rlabel_position(0)
        # matplotlib.pyplot.axis((0, math.pi * 2, 0, radiusMax))

        # # set radius grids to 15, 30, etc, which means 6 numbers (r=0 not included)
        # radiusGridAngles = numpy.arange(15, 91, 15)
        # radiusGridValues = []
        # for angle in radiusGridAngles:
        #     #                        - project the 15, 30, 45 as spherical coords, and select the r part of theta r-
        #     #               - append to list of radii -

        #     radiusGridValues.append(projectOrientations3d([0, angle * math.pi / 180.0, 1], "spherical", projection)[1][1])
        # #                                       --- list comprehension to print 15°, 30°, 45° ----------
        # # ax.set_rgrids(radiusGridValues, labels=[r"%02i$^\circ$" % (x) for x in numpy.arange(15, 91, 15)], angle=None, fmt=None)
        
        # ax.plot(projection_theta_r[:, 0], projection_theta_r[:, 1], ".", markersize=pointMarkerSize)

        ax.plot(projection_xy[:, 0], projection_xy[:, 1], ".", markersize=pointMarkerSize)

        if show:
            if plot == "points":
                matplotlib.pyplot.show()

    if plot == "bins" or plot == "both":
        # ========================================================================
        # ==== Binning the data -- this could be optional...                   ===
        # ========================================================================
        # This code inspired from Hugues Talbot and Clara Jaquet's developments.
        # As published in:
        #   Identifying and following particle-to-particle contacts in real granular media: an experimental challenge
        #   Gioacchino Viggiani, Edward Andò, Clara Jaquet and Hugues Talbot
        #   Keynote Lecture
        #   Particles and Grains 2013 Sydney
        #
        # ...The number of radial bins (numberOfRings)
        # defines the radial binning, and for each radial bin starting from the centre,
        # the number of angular bins is  4(2n + 1)
        #
        import matplotlib.collections

        # from matplotlib.colors import Normalize
        import matplotlib.colorbar
        import matplotlib.patches

        if plot == "both":
            ax = fig.add_subplot(122, polar=True)
        if plot == "bins":
            fig = matplotlib.pyplot.figure()
            ax = fig.add_subplot(111, polar=True)

        if VERBOSE:
            print("\t-> Starting Data binning...")

        # This must be an integer -- could well be a parameter if this becomes a function.
        if VERBOSE:
            print("\t-> Number of Rings (radial bins) = ", numberOfRings)

        # As per the publication, the maximum number of bins for each ring, coming from the inside out is 4(2n + 1):
        numberOfAngularBinsPerRing = numpy.arange(1, numberOfRings + 1, 1)
        numberOfAngularBinsPerRing = 4 * (2 * numberOfAngularBinsPerRing - 1)

        if VERBOSE:
            print("\t-> Number of angular bins per ring = ", numberOfAngularBinsPerRing)

        # defining an array with dimensions numberOfRings x numberOfAngularBinsPerRing
        binCounts = numpy.zeros((numberOfRings, numberOfAngularBinsPerRing[-1]))

        # ========================================================================
        # ==== Start counting the vectors into bins                            ===
        # ========================================================================
        for vectorN in range(numberOfPoints):
            # unpack projected angle and radius for this point
            angle, radius = projection_theta_r[vectorN, :]

            # Flip over negative angles
            if angle < 0:
                angle += 2 * math.pi
            if angle > 2 * math.pi:
                angle -= 2 * math.pi

            # Calculate right ring number
            ringNumber = int(numpy.floor(radius / (radiusMax / float(numberOfRings))))

            # Check for overflow
            if ringNumber > numberOfRings - 1:
                if VERBOSE:
                    print("\t-> Point with projected radius = {:f} is a problem (radiusMax = {:f}), putting in furthest  bin".format(radius, radiusMax))
                ringNumber = numberOfRings - 1

            # Calculate the angular bin
            angularBin = int(numpy.floor((angle) / (2 * math.pi / float(numberOfAngularBinsPerRing[ringNumber])))) + 1

            # print "numberOfAngularBinsPerRing", numberOfAngularBinsPerRing[ringNumber] - 1
            # Check for overflow
            #  in case it doesn't belong in the last angularBin, it has to be put in the first one!
            if angularBin > numberOfAngularBinsPerRing[ringNumber] - 1:
                if VERBOSE:
                    print("\t-> Point with projected angle = %f does not belong to the last bin, putting in first bin" % (angle))
                angularBin = 0

            # now that we know what ring, and angular bin you're in add one count!
            binCounts[ringNumber, angularBin] += 1

        # ========================================================================
        # === Plotting binned data                                             ===
        # ========================================================================

        plottingRadii = numpy.linspace(radiusMax / float(numberOfRings), radiusMax, numberOfRings)
        # print "Plotting radii:", plottingRadii

        # ax  = fig.add_subplot(122, polar=True)
        # matplotlib.pyplot.axis()
        # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        bars = []

        # add two fake, small circles at the beginning so that they are overwritten
        #   they will be coloured with the min and max colour
        #              theta   radius    width
        bars.append([0, radiusMax, 2 * math.pi])
        bars.append([0, radiusMax, 2 * math.pi])
        # bars.append(ax.bar(0,   radiusMax,    2*math.pi, bottom=0.0))
        # bars.append(ax.bar(0,   radiusMax,    2*math.pi, bottom=0.0))

        # --- flatifiying binned data for colouring wedges                    ===
        flatBinCounts = numpy.zeros(numpy.sum(numberOfAngularBinsPerRing) + 2)

        # Bin number as we go through the bins to add the counts in order to the flatBinCounts
        # This is two in order to skip the first to fake bins which set the colour bar.
        binNumber = 2

        # --- Plotting binned data, from the outside, inwards.                 ===
        if binNormalisation:
            avg_binCount = float(numberOfPoints) / numpy.sum(numberOfAngularBinsPerRing)
            # print "\t-> Number of points = ", numberOfPoints
            # print "\t-> Number of bins   = ", numpy.sum(numberOfAngularBinsPerRing)
            if VERBOSE:
                print("\t-> Average binCount = ", avg_binCount)

        for ringNumber in range(numberOfRings)[::-1]:
            deltaTheta = 360 / float(numberOfAngularBinsPerRing[ringNumber])
            deltaThetaRad = 2 * math.pi / float(numberOfAngularBinsPerRing[ringNumber])

            # --- Angular bins                                                 ---
            for angularBin in range(numberOfAngularBinsPerRing[ringNumber]):
                # ...or add bars
                #                           theta                             radius                  width
                bars.append([angularBin * deltaThetaRad - deltaThetaRad / 2.0, plottingRadii[ringNumber], deltaThetaRad])
                # bars.append(ax.bar(angularBin*deltaThetaRad - deltaThetaRad/2.0, plottingRadii[ ringNumber ], deltaThetaRad, bottom=0.0))

                # Add the number of vectors counted for this bin
                if binNormalisation:
                    flatBinCounts[binNumber] = binCounts[ringNumber, angularBin] / avg_binCount
                else:
                    flatBinCounts[binNumber] = binCounts[ringNumber, angularBin]

                # Add one to bin number
                binNumber += 1

        del binNumber

        # figure out auto values if they're requested.
        if binValueMin is None:
            binValueMin = flatBinCounts[2::].min()
        if binValueMax is None:
            binValueMax = flatBinCounts[2::].max()

        # Add two flat values for the initial wedges.
        flatBinCounts[0] = binValueMin
        flatBinCounts[1] = binValueMax

        ##                           theta                   radius                          width
        barsPlot = ax.bar(numpy.array(bars)[:, 0], numpy.array(bars)[:, 1], width=numpy.array(bars)[:, 2], bottom=0.0)

        for binCount, bar in zip(flatBinCounts, barsPlot):
            bar.set_facecolor(cmap((binCount - binValueMin) / float(binValueMax - binValueMin)))

        # matplotlib.pyplot.axis([ 0, radiusMax, 0, radiusMax ])
        matplotlib.pyplot.axis([0, numpy.deg2rad(360), 0, radiusMax])

        # colorbar = matplotlib.pyplot.colorbar(barsPlot, norm=matplotlib.colors.Normalize(vmin=minBinValue, vmax=maxBinValue))
        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.

        norm = matplotlib.colors.Normalize(vmin=binValueMin, vmax=binValueMax)

        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        ax3 = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        cb1 = matplotlib.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm, label="Number of vectors in bin")

        # set the line along which the numbers are plotted to 0°
        # ax.set_rlabel_position(0)

        # set radius grids to 15, 30, etc, which means 6 numbers (r=0 not included)
        radiusGridAngles = numpy.arange(15, 91, 15)
        radiusGridValues = []
        for angle in radiusGridAngles:
            #                        - project the 15, 30, 45 as spherical coords, and select the r part of theta r-
            #               - append to list of radii -
            radiusGridValues.append(projectOrientations3d([0, angle * math.pi / 180.0, 1], "spherical", projection)[1][1])
        #                                       --- list comprehension to print 15°, 30°, 45° ----------
        ax.set_rgrids(radiusGridValues, labels=[r"%02i$^\circ$" % (x) for x in numpy.arange(15, 91, 15)], angle=None, fmt=None)

        fig.subplots_adjust(left=0.05, right=0.85)
        # cb1.set_label('Some Units')

        if saveFigPath is not None:
            matplotlib.pyplot.savefig(saveFigPath)
            matplotlib.pyplot.close()
        else:
            matplotlib.pyplot.show()