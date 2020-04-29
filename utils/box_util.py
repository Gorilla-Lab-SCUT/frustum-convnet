""" Helper functions for calculating 2D and 3D bounding box IoU.

Collected by Charles R. Qi
Date: September 2017
"""

import numpy as np

from scipy.spatial import ConvexHull


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """
    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)


def poly_area(x, y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :])**2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :])**2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :])**2))
    return a * b * c


def is_clockwise(p):
    x = p[:, 0]
    y = p[:, 1]
    return np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)) > 0


def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def box3d_iou_pair(corners1, corners2):
    ''' Compute paired 3D bounding box IoU.
    /*
      camera coordinate
                7 -------- 4
               /|         /|
              6 -------- 5 .
              | |        | |
              . 3 -------- 0
              |/         |/
              2 -------- 1
     */
    Input:
        corners1: numpy array (n,8,3), assume up direction is negative Y
        corners2: numpy array (n,8,3), assume up direction is negative Y
    Output:
        ious: numpy array (n, 2)
        [:, 0] BEV box IoU
        [:, 1] 3D bounding box IoU
    '''
    # corner points are in counter clockwise order
    assert len(corners1) == len(corners2), "number of boxes sholud be equal"
    num_pairs = len(corners1)
    ious = np.zeros((num_pairs, 2), dtype=corners1.dtype)
    for i in range(num_pairs):
        iou_3d, iou_2d = box3d_iou(corners1[i], corners2[i])
        ious[i, 0] = iou_2d
        ious[i, 1] = iou_3d

    return ious


if __name__ == '__main__':

    # Function for polygon ploting
    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt

    def plot_polys(plist, scale=500.0):
        fig, ax = plt.subplots()
        patches = []
        for p in plist:
            poly = Polygon(np.array(p) / scale, True)
            patches.append(poly)

    # pc = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.5)
    # colors = 100 * np.random.rand(len(patches))
    # pc.set_array(np.array(colors))
    # ax.add_collection(pc)
    # plt.show()

    # Demo on ConvexHull
    points = np.random.rand(30, 2)  # 30 random points in 2-D
    hull = ConvexHull(points)
    # **In 2D "volume" is is area, "area" is perimeter
    print('Hull area: ', hull.volume)
    for simplex in hull.simplices:
        print(simplex)

    # Demo on convex hull overlaps
    sub_poly = [(0, 0), (300, 0), (300, 300), (0, 300)]
    clip_poly = [(150, 150), (300, 300), (150, 450), (0, 300)]
    inter_poly = polygon_clip(sub_poly, clip_poly)
    print(poly_area(np.array(inter_poly)[:, 0], np.array(inter_poly)[:, 1]))

    # Test convex hull interaction function
    rect1 = [(50, 0), (50, 300), (300, 300), (300, 0)]
    rect2 = [(150, 150), (300, 300), (150, 450), (0, 300)]
    plot_polys([rect1, rect2])
    inter, area = convex_hull_intersection(rect1, rect2)
    print(inter, area)
    if inter is not None:
        print(poly_area(np.array(inter)[:, 0], np.array(inter)[:, 1]))

    print('------------------')
    rect1 = [(0.30026005199835404, 8.9408694211408424),
             (-1.1571105364358421, 9.4686676477075533),
             (0.1777082043006144, 13.154404877812102),
             (1.6350787927348105, 12.626606651245391)]
    rect1 = [rect1[0], rect1[3], rect1[2], rect1[1]]
    rect2 = [(0.23908745901608636, 8.8551095691132886),
             (-1.2771419487733995, 9.4269062966181956),
             (0.13138836963152717, 13.161896351296868),
             (1.647617777421013, 12.590099623791961)]
    rect2 = [rect2[0], rect2[3], rect2[2], rect2[1]]
    plot_polys([rect1, rect2])
    inter, area = convex_hull_intersection(rect1, rect2)
    print(inter, area)
