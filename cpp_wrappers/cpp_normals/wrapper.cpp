#include <Python.h>
#include <numpy/arrayobject.h>
#include "neighbors/neighbors.h"
#include <string>



// docstrings for our module
// *************************

static char module_docstring[] = "This module provides two methods to compute radius neighbors from pointclouds or batch of pointclouds";

static char batch_query_docstring[] = "Method to get radius neighbors in a batch of stacked pointclouds";


// Declare the functions
// *********************

static PyObject *batch_neighbors(PyObject *self, PyObject *args, PyObject *keywds);


// Specify the members of the module
// *********************************

static PyMethodDef module_methods[] = 
{
	{ "batch_query", (PyCFunction)batch_neighbors, METH_VARARGS | METH_KEYWORDS, batch_query_docstring },
	{NULL, NULL, 0, NULL}
};


// Initialize the module
// *********************

static struct PyModuleDef moduledef = 
{
    PyModuleDef_HEAD_INIT,
    "radius_neighbors",		// m_name
    module_docstring,       // m_doc
    -1,                     // m_size
    module_methods,         // m_methods
    NULL,                   // m_reload
    NULL,                   // m_traverse
    NULL,                   // m_clear
    NULL,                   // m_free
};

PyMODINIT_FUNC PyInit_radius_neighbors(void)
{
    import_array();
	return PyModule_Create(&moduledef);
}


// Definition of main method
// **********************************

static PyObject* batch_neighbors(PyObject* self, PyObject* args, PyObject* keywds)
{

	// Manage inputs
	// *************

	// Args containers
	PyObject* points_obj = NULL;
	PyObject* normals_obj = NULL;
	PyObject* sigma_s_obj = NULL;
	PyObject* sigma_r_obj = NULL;

	// Keywords containers
	static char* kwlist[] = { "points", "normals", "sigma_s", "sigma_r", "radius", "self_included", NULL };
	float radius = 0.1;
	int self_included = 0;

	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO|fi", kwlist, &points_obj, &normals_obj, &sigma_s_obj, &sigma_r_obj, &radius, &self_included))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}


	// Interpret the input objects as numpy arrays.
	PyObject* points_array = PyArray_FROM_OTF(points_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* normals_array = PyArray_FROM_OTF(normals_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* sigma_s_array = PyArray_FROM_OTF(sigma_s_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject* sigma_r_array = PyArray_FROM_OTF(sigma_r_obj, NPY_FLOAT, NPY_IN_ARRAY);


	// Check Data
	// **********

	// Verify data was load correctly.
	if (points_array == NULL)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(normals_array);
		Py_XDECREF(sigma_s_array);
		Py_XDECREF(sigma_r_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting *points* to numpy arrays of type float32");
		return NULL;
	}
	if (normals_array == NULL)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(normals_array);
		Py_XDECREF(sigma_s_array);
		Py_XDECREF(sigma_r_array);
		PyErr_SetString(PyExc_RuntimeError, "Error converting *normals* to numpy arrays of type float32");
		return NULL;
	}

	// Check that the input array respect the dims
	if ((int)PyArray_NDIM(points_array) != 2 || (int)PyArray_DIM(points_array, 1) != 3)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(normals_array);
		Py_XDECREF(sigma_s_array);
		Py_XDECREF(sigma_r_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions: points.shape is not (N, 3)");
		return NULL;
	}
	if ((int)PyArray_NDIM(normals_array) != 2 || (int)PyArray_DIM(normals_array, 1) != 3)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(normals_array);
		Py_XDECREF(sigma_s_array);
		Py_XDECREF(sigma_r_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions: normals.shape is not (N, 3)");
		return NULL;
	}
	if ((int)PyArray_DIM(points_array, 0) != (int)PyArray_DIM(normals_array, 0))
	{
		Py_XDECREF(points_array);
		Py_XDECREF(normals_array);
		Py_XDECREF(sigma_s_array);
		Py_XDECREF(sigma_r_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions: points.shape[0] != normals.shape[0], different point numbers");
		return NULL;
	}

	// Check 1-d array 
	if ((int)PyArray_NDIM(sigma_s_array) != 1 || (int)PyArray_DIM(sigma_s_array, 0) < 1)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(normals_array);
		Py_XDECREF(sigma_s_array);
		Py_XDECREF(sigma_r_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions: sigma_s.shape is not (s,)");
		return NULL;
	}
	if ((int)PyArray_NDIM(sigma_r_array) != 1 || (int)PyArray_DIM(sigma_r_array, 0) < 1)
	{
		Py_XDECREF(points_array);
		Py_XDECREF(normals_array);
		Py_XDECREF(sigma_s_array);
		Py_XDECREF(sigma_r_array);
		PyErr_SetString(PyExc_RuntimeError, "Wrong dimensions: sigma_r.shape is not (r,)");
		return NULL;
	}


	// Main Methods
	// ************

	// Number of points
	int n_points = (int)PyArray_DIM(points_array, 0);
	int n_normals = (int)PyArray_DIM(normals_array, 0);
	int ns = (int)PyArray_DIM(sigma_s_array, 0);
	int nr = (int)PyArray_DIM(sigma_r_array, 0);

	// Call the C++ function
	// *********************

	// Convert PyArray to Cloud C++ class
	vector<PointXYZ> points;
	vector<PointXYZ> normals;
	points = vector<PointXYZ>((PointXYZ*)PyArray_DATA(points_array), (PointXYZ*)PyArray_DATA(points_array) + n_points);
	normals = vector<PointXYZ>((PointXYZ*)PyArray_DATA(normals_array), (PointXYZ*)PyArray_DATA(normals_array) + n_normals);

	vector<float> sigma_s;
	vector<float> sigma_r;
	sigma_s = vector<float>((float*)PyArray_DATA(sigma_s_array), (float*)PyArray_DATA(sigma_s_array) + ns);
	sigma_r = vector<float>((float*)PyArray_DATA(sigma_r_array), (float*)PyArray_DATA(sigma_r_array) + nr);

	// Create result containers
	vector<float> ret_normals;
	vector<float> rotation;

	// Compute normal features
	normal_filtering_multiscale(points, normals, sigma_s, sigma_r, radius, self_included, ret_normals);

	// Check result
	if (ret_normals.size() < 1)
	{
		PyErr_SetString(PyExc_RuntimeError, "Error");
		return NULL;
	}


	// Manage outputs
	// **************

	// Dimension of output containers
	int feat_dim = 3*ns*nr + 3*self_included;
	npy_intp* output_dims = new npy_intp[2];
	output_dims[0] = n_points;
	output_dims[1] = feat_dim;

	// Create output array
	PyObject* res_obj = PyArray_SimpleNew(2, output_dims, NPY_FLOAT);
	PyObject* ret = NULL;

	// Fill output array with values
	size_t size_in_bytes = n_points * feat_dim * sizeof(float);
	memcpy(PyArray_DATA(res_obj), ret_normals.data(), size_in_bytes);

	// Merge results
	ret = Py_BuildValue("N", res_obj);

	// Clean up
	// ********

	Py_XDECREF(points_array);
	Py_XDECREF(normals_array);
	Py_XDECREF(sigma_s_array);
	Py_XDECREF(sigma_r_array);

	return ret;
}
