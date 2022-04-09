#include <Python.h>
#include <numpy/arrayobject.h>
#include "src/heights.h"
#include <string>



// docstrings for our module
// *************************

static char module_docstring[] = "This module implements the function to compute height maps from an input point cloud";

static char function_docstring[] = "Function to get height distances in a neighborhood of each point";


// Declare the functions
// *********************

static PyObject *main_function(PyObject *self, PyObject *args, PyObject *keywds);


// Specify the members of the module
// *********************************

static PyMethodDef module_methods[] = 
{
	{ "height_distances", (PyCFunction)main_function, METH_VARARGS | METH_KEYWORDS, function_docstring },
	{NULL, NULL, 0, NULL}
};


// Initialize the module
// *********************

static struct PyModuleDef moduledef = 
{
    PyModuleDef_HEAD_INIT,
    "cpp_height",			// m_name
    module_docstring,       // m_doc
    -1,                     // m_size
    module_methods,         // m_methods
    NULL,                   // m_reload
    NULL,                   // m_traverse
    NULL,                   // m_clear
    NULL,                   // m_free
};

PyMODINIT_FUNC PyInit_cpp_height(void)
{
    import_array();
	return PyModule_Create(&moduledef);
}


// Definition of main method
// **********************************

static PyObject* main_function(PyObject* self, PyObject* args, PyObject* keywds)
{

	// Manage inputs
	// *************

	// Args containers
	PyObject* points_obj = NULL;
	PyObject* features_obj = NULL;
	PyObject* rotation_obj = NULL;

	// Keywords containers
	static char* kwlist[] = { "points", "features", "rotation", "map_size", "query_k", NULL };
	int map_size = 0, query_k = 0;

	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOO|ii", kwlist, &points_obj, &features_obj, &rotation_obj, &map_size, &query_k))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}


	// Interpret the input objects as numpy arrays.
	PyObject* points_array = PyArray_FROM_OTF(points_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* features_array = PyArray_FROM_OTF(features_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* rotation_array = PyArray_FROM_OTF(rotation_obj, NPY_FLOAT, NPY_IN_ARRAY);


	// Check Data
	// **********

	// Verify data was load correctly.
	if (points_array == NULL)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(features_array);
		Py_XDECREF(rotation_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting *points* to numpy arrays of type float32");
		return NULL;
	}
	if (features_array == NULL)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(features_array);
		Py_XDECREF(rotation_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting *normals* to numpy arrays of type float32");
		return NULL;
	}
	if (rotation_array == NULL)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(features_array);
		Py_XDECREF(rotation_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting *rotation* to numpy arrays of type float32");
		return NULL;
	}

	// Check that the input array respect the dims
	if ((int)PyArray_NDIM(points_array) != 2 || (int)PyArray_DIM(points_array, 1) != 3)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(features_array);
		Py_XDECREF(rotation_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions: points.shape is not (N, 3)");
		return NULL;
	}
	if ((int)PyArray_NDIM(features_array) != 2)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(features_array);
		Py_XDECREF(rotation_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions: normals.shape is not (N, f)");
		return NULL;
	}
	if ((int)PyArray_NDIM(rotation_array) != 3 || (int)PyArray_DIM(rotation_array, 1) != 3 || (int)PyArray_DIM(rotation_array, 2) != 3)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(features_array);
		Py_XDECREF(rotation_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions: rotation.shape is not (N, 3, 3)");
		return NULL;
	}
	if ((int)PyArray_DIM(points_array, 0) != (int)PyArray_DIM(features_array, 0) || (int)PyArray_DIM(points_array, 0) != (int)PyArray_DIM(rotation_array, 0))
	{
		Py_XDECREF(points_array);
		Py_XDECREF(features_array);
		Py_XDECREF(rotation_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions: array shape[0] is not consistent");
		return NULL;
	}


	// Main Methods
	// ************

	// Number of points
	int n_points = (int)PyArray_DIM(points_array, 0);
	int n_normals = (int)PyArray_DIM(features_array, 0);
	int n_features = (int)PyArray_DIM(features_array, 1) / 3;

	// Call the C++ function
	// *********************

	// Convert PyArray to Cloud C++ class
	vector<PointXYZ> points;
	points = vector<PointXYZ>((PointXYZ*)PyArray_DATA(points_array), (PointXYZ*)PyArray_DATA(points_array) + n_points);

	// Convert PyArray to float array
	vector<float> features;
	features = vector<float>((float*)PyArray_DATA(features_array), (float*)PyArray_DATA(features_array) + n_normals*n_features*3);

	// Convert PyArray to 3-d float array
	vector<float> rotation;
	rotation = vector<float>((float*)PyArray_DATA(rotation_array), (float*)PyArray_DATA(rotation_array) + n_points*3*3);


	// Create result containers
	vector<float> ret_data;

	// Compute normal features
	compute_height_map(points, features, rotation, n_features, map_size, query_k, ret_data);

	// Check result
	if (ret_data.size() < 1)
	{
		PyErr_SetString(PyExc_RuntimeError, "Error");
		return NULL;
	}


	// Manage outputs
	// **************

	// Dimension of output containers
	npy_intp* output_dims = new npy_intp[4];
	output_dims[0] = n_points;
	output_dims[1] = n_features;
	output_dims[2] = map_size;
	output_dims[3] = map_size;

	// Create output array
	PyObject* res_obj = PyArray_SimpleNew(4, output_dims, NPY_FLOAT);
	PyObject* ret = NULL;

	// Fill output array with values
	size_t size_in_bytes = n_points * n_features * map_size * map_size * sizeof(float);
	memcpy(PyArray_DATA(res_obj), ret_data.data(), size_in_bytes);

	// Merge results
	ret = Py_BuildValue("N", res_obj);

	// Clean up
	// ********

	Py_XDECREF(points_array);
	Py_XDECREF(features_array);
	Py_XDECREF(rotation_array);

	return ret;
}
