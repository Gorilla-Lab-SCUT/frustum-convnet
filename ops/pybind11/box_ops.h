#ifndef BOX_OPS_H
#define BOX_OPS_H

#include <math.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <boost/geometry.hpp>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace pybind11::literals;

template<typename DType, typename ShapeContainer>
inline py::array_t<DType> constant(ShapeContainer shape, DType value){
    // create ROWMAJOR array.
    py::array_t<DType> array(shape);
    std::fill(array.mutable_data(), array.mutable_data() + array.size(), value);
    return array;
}

template<typename DType>
inline py::array_t<DType> zeros(std::vector<long int> shape){
    return constant<DType, std::vector<long int>>(shape, 0);
}
template <typename DType>
py::array_t<DType>
rbbox_iou(py::array_t<DType> box_corners, py::array_t<DType> qbox_corners,
          py::array_t<DType> standup_iou, DType standup_thresh) {
  namespace bg = boost::geometry;
  typedef bg::model::point<DType, 2, bg::cs::cartesian> point_t;
  typedef bg::model::polygon<point_t> polygon_t;
  polygon_t poly, qpoly;
  std::vector<polygon_t> poly_inter, poly_union;
  DType inter_area, union_area;
  auto box_corners_r = box_corners.template unchecked<3>();
  auto qbox_corners_r = qbox_corners.template unchecked<3>();
  auto standup_iou_r = standup_iou.template unchecked<2>();
  auto N = box_corners_r.shape(0);
  auto K = qbox_corners_r.shape(0);
  py::array_t<DType> overlaps = zeros<DType>({N, K});
  auto overlaps_rw = overlaps.template mutable_unchecked<2>();
  if (N == 0 || K == 0) {
    return overlaps;
  }
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      if (standup_iou_r(n, k) <= standup_thresh)
        continue;
      bg::append(poly, point_t(box_corners_r(n, 0, 0), box_corners_r(n, 0, 1)));
      bg::append(poly, point_t(box_corners_r(n, 1, 0), box_corners_r(n, 1, 1)));
      bg::append(poly, point_t(box_corners_r(n, 2, 0), box_corners_r(n, 2, 1)));
      bg::append(poly, point_t(box_corners_r(n, 3, 0), box_corners_r(n, 3, 1)));
      bg::append(poly, point_t(box_corners_r(n, 0, 0), box_corners_r(n, 0, 1)));
      bg::append(qpoly,
                 point_t(qbox_corners_r(k, 0, 0), qbox_corners_r(k, 0, 1)));
      bg::append(qpoly,
                 point_t(qbox_corners_r(k, 1, 0), qbox_corners_r(k, 1, 1)));
      bg::append(qpoly,
                 point_t(qbox_corners_r(k, 2, 0), qbox_corners_r(k, 2, 1)));
      bg::append(qpoly,
                 point_t(qbox_corners_r(k, 3, 0), qbox_corners_r(k, 3, 1)));
      bg::append(qpoly,
                 point_t(qbox_corners_r(k, 0, 0), qbox_corners_r(k, 0, 1)));

      bg::intersection(poly, qpoly, poly_inter);

      if (!poly_inter.empty()) {
        inter_area = bg::area(poly_inter.front());
        bg::union_(poly, qpoly, poly_union);
        if (!poly_union.empty()) {
          union_area = bg::area(poly_union.front());
          overlaps_rw(n, k) = inter_area / union_area;
        }
        poly_union.clear();
      }
      poly.clear();
      qpoly.clear();
      poly_inter.clear();
    }
  }
  return overlaps;
}

template <typename DType>
py::array_t<DType>
rbbox_iou_3d(py::array_t<DType> box_corners, py::array_t<DType> qbox_corners,
          py::array_t<DType> standup_iou, DType standup_thresh) {
  namespace bg = boost::geometry;
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

  typedef bg::model::point<DType, 2, bg::cs::cartesian> point_t;
  typedef bg::model::polygon<point_t> polygon_t;
  polygon_t poly, qpoly;
  std::vector<polygon_t> poly_inter, poly_union;
  
  DType inter_area, inter_vol;
  DType ymin, ymax;
  DType area, qarea, h, qh;
  DType vol, qvol;
  DType zero = 0.0;
  auto box_corners_r = box_corners.template unchecked<3>();
  auto qbox_corners_r = qbox_corners.template unchecked<3>();
  auto standup_iou_r = standup_iou.template unchecked<2>();
  auto N = box_corners_r.shape(0);
  auto K = qbox_corners_r.shape(0);
  py::array_t<DType> overlaps = zeros<DType>({N, K});
  auto overlaps_rw = overlaps.template mutable_unchecked<2>();
  if (N == 0 || K == 0) {
    return overlaps;
  }
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      if (standup_iou_r(n, k) <= standup_thresh)
        continue;
      bg::append(poly, point_t(box_corners_r(n, 6, 0), box_corners_r(n, 6, 2)));
      bg::append(poly, point_t(box_corners_r(n, 7, 0), box_corners_r(n, 7, 2)));
      bg::append(poly, point_t(box_corners_r(n, 4, 0), box_corners_r(n, 4, 2)));
      bg::append(poly, point_t(box_corners_r(n, 5, 0), box_corners_r(n, 5, 2)));
      bg::append(poly, point_t(box_corners_r(n, 6, 0), box_corners_r(n, 6, 2)));
      bg::append(qpoly,
                 point_t(qbox_corners_r(k, 6, 0), qbox_corners_r(k, 6, 2)));
      bg::append(qpoly,
                 point_t(qbox_corners_r(k, 7, 0), qbox_corners_r(k, 7, 2)));
      bg::append(qpoly,
                 point_t(qbox_corners_r(k, 4, 0), qbox_corners_r(k, 4, 2)));
      bg::append(qpoly,
                 point_t(qbox_corners_r(k, 5, 0), qbox_corners_r(k, 5, 2)));
      bg::append(qpoly,
                 point_t(qbox_corners_r(k, 6, 0), qbox_corners_r(k, 6, 2)));

      bg::intersection(poly, qpoly, poly_inter);

      if (!poly_inter.empty()) {
        inter_area = bg::area(poly_inter.front());
        bg::union_(poly, qpoly, poly_union);
        if (!poly_union.empty()) {
          // union_area = bg::area(poly_union.front());

          ymax = std::min(box_corners_r(n, 0, 1), qbox_corners_r(k, 0, 1));
          ymin = std::max(box_corners_r(n, 4, 1), qbox_corners_r(k, 4, 1));

          h = box_corners_r(n, 0, 1) - box_corners_r(n, 4, 1);
          qh = qbox_corners_r(k, 0, 1) - qbox_corners_r(k, 4, 1);

          area = bg::area(poly);
          qarea = bg::area(qpoly);

          inter_vol = inter_area * std::max(zero, ymax - ymin);;

          vol = std::max(zero, area * h);
          qvol = std::max(zero, qarea * qh);

          overlaps_rw(n, k) = inter_vol / (vol + qvol - inter_vol);
        }
        poly_union.clear();
      }
      poly.clear();
      qpoly.clear();
      poly_inter.clear();
    }
  }
  return overlaps;
}

template <typename DType>
py::array_t<DType>
rbbox_iou_3d_pair(py::array_t<DType> box_corners, py::array_t<DType> qbox_corners) {
  namespace bg = boost::geometry;
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

  typedef bg::model::point<DType, 2, bg::cs::cartesian> point_t;
  typedef bg::model::polygon<point_t> polygon_t;
  polygon_t poly, qpoly;
  std::vector<polygon_t> poly_inter, poly_union;
  
  DType inter_area, union_area, inter_vol;
  DType ymin, ymax;
  DType area, qarea, h, qh;
  DType vol, qvol;
  DType zero = 0.0;
  auto box_corners_r = box_corners.template unchecked<3>();
  auto qbox_corners_r = qbox_corners.template unchecked<3>();
  auto N = box_corners_r.shape(0);
  auto K = qbox_corners_r.shape(0);
  
  py::array_t<DType> overlaps = zeros<DType>({N, 2});
  auto overlaps_rw = overlaps.template mutable_unchecked<2>();
  if (N == 0 || K == 0 || N != K) {
    return overlaps;
  }
  for (int n = 0; n < N; ++n) {
   
    bg::append(poly, point_t(box_corners_r(n, 6, 0), box_corners_r(n, 6, 2)));
    bg::append(poly, point_t(box_corners_r(n, 7, 0), box_corners_r(n, 7, 2)));
    bg::append(poly, point_t(box_corners_r(n, 4, 0), box_corners_r(n, 4, 2)));
    bg::append(poly, point_t(box_corners_r(n, 5, 0), box_corners_r(n, 5, 2)));
    bg::append(poly, point_t(box_corners_r(n, 6, 0), box_corners_r(n, 6, 2)));
    bg::append(qpoly,
               point_t(qbox_corners_r(n, 6, 0), qbox_corners_r(n, 6, 2)));
    bg::append(qpoly,
               point_t(qbox_corners_r(n, 7, 0), qbox_corners_r(n, 7, 2)));
    bg::append(qpoly,
               point_t(qbox_corners_r(n, 4, 0), qbox_corners_r(n, 4, 2)));
    bg::append(qpoly,
               point_t(qbox_corners_r(n, 5, 0), qbox_corners_r(n, 5, 2)));
    bg::append(qpoly,
               point_t(qbox_corners_r(n, 6, 0), qbox_corners_r(n, 6, 2)));

    bg::intersection(poly, qpoly, poly_inter);

    if (!poly_inter.empty()) {
      inter_area = bg::area(poly_inter.front());
      bg::union_(poly, qpoly, poly_union);
      if (!poly_union.empty()) {
        union_area = bg::area(poly_union.front());

        ymax = std::min(box_corners_r(n, 0, 1), qbox_corners_r(n, 0, 1));
        ymin = std::max(box_corners_r(n, 4, 1), qbox_corners_r(n, 4, 1));

        h = box_corners_r(n, 0, 1) - box_corners_r(n, 4, 1);
        qh = qbox_corners_r(n, 0, 1) - qbox_corners_r(n, 4, 1);

        area = bg::area(poly);
        qarea = bg::area(qpoly);

        inter_vol = inter_area * std::max(zero, ymax - ymin);;

        vol = std::max(zero, area * h);
        qvol = std::max(zero, qarea * qh);

        overlaps_rw(n, 0) = inter_area / union_area;
        overlaps_rw(n, 1) = inter_vol / (vol + qvol - inter_vol);

      }
      poly_union.clear();
    }
    poly.clear();
    qpoly.clear();
    poly_inter.clear();
  }
  
  return overlaps;
}

#endif