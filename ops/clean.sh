set -x 
set -e

cd query_depth_point
rm -f *.so
python setup.py clean --all
cd ..

cd pybind11
python setup.py clean --all
rm -f *.so

