#include "box_ops.h"

PYBIND11_MODULE(box_ops_cc, m) {
  m.doc() = "box ops written by c++";
  m.def("rbbox_iou", &rbbox_iou<double>,
        py::return_value_policy::reference_internal, "rbbox iou",
        "box_corners"_a = 1, "qbox_corners"_a = 2, "standup_iou"_a = 3,
        "standup_thresh"_a = 4);
  m.def("rbbox_iou", &rbbox_iou<float>,
        py::return_value_policy::reference_internal, "rbbox iou",
        "box_corners"_a = 1, "qbox_corners"_a = 2, "standup_iou"_a = 3,
        "standup_thresh"_a = 4);

  m.def("rbbox_iou_3d", &rbbox_iou_3d<double>,
        py::return_value_policy::reference_internal, "rbbox iou 3d",
        "box_corners"_a = 1, "qbox_corners"_a = 2, "standup_iou"_a = 3,
        "standup_thresh"_a = 4);

  m.def("rbbox_iou_3d", &rbbox_iou_3d<float>,
        py::return_value_policy::reference_internal, "rbbox iou 3d",
        "box_corners"_a = 1, "qbox_corners"_a = 2, "standup_iou"_a = 3,
        "standup_thresh"_a = 4);

  m.def("rbbox_iou_3d_pair", &rbbox_iou_3d_pair<double>,
        py::return_value_policy::reference_internal, "rbbox iou 3d pair",
        "box_corners"_a = 1, "qbox_corners"_a = 2);

  m.def("rbbox_iou_3d_pair", &rbbox_iou_3d_pair<float>,
        py::return_value_policy::reference_internal, "rbbox iou 3d pair",
        "box_corners"_a = 1, "qbox_corners"_a = 2);

}
