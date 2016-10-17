#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <Eigen/Dense>
#include <flann/flann.hpp>

#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/Exporter.hpp>      // C++ exporter interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing fla

template<typename Type, int Rows = 3, int Cols = 1>
static bool import_obj(const std::string& filename, std::vector<Eigen::Matrix<Type, Rows, Cols>>& points3D)
{
	std::ifstream inFile;
	inFile.open(filename);

	if (!inFile.is_open())
	{
		std::cerr << "Error: Could not open obj input file: " << filename << std::endl;
		return false;
	}

	points3D.clear();

	int i = 0;
	while (inFile)
	{
		std::string str;

		if (!std::getline(inFile, str))
		{
			if (inFile.eof())
				return true;

			std::cerr << "Error: Problems when reading obj file: " << filename << std::endl;
			return false;
		}

		if (str[0] == 'v')
		{
			std::stringstream ss(str);
			std::vector <std::string> record;

			char c;
			Type x, y, z;
			ss >> c >> x >> y >> z;

			const Eigen::Matrix<Type, 3, 1> p(x, y, z);
			points3D.push_back(p);
		}
	}

	inFile.close();

	return true;
}


template<typename Type>
static bool import_obj(const std::string& filename, std::vector<Type>& vertices, std::vector<Type>& normals)
{
	std::ifstream inFile;
	inFile.open(filename);

	if (!inFile.is_open())
	{
		std::cerr << "Error: Could not open obj input file: " << filename << std::endl;
		return false;
	}

	vertices.clear();
	normals.clear();

	int i = 0;
	while (inFile)
	{
		std::string str;

		if (!std::getline(inFile, str))
		{
			if (inFile.eof())
				return true;

			std::cerr << "Error: Problems when reading obj file: " << filename << std::endl;
			return false;
		}

		//if (str[0] == 'v' && str[1] == ' ')
		if (str[0] == 'v')
		{
			if (str[1] == ' ')  // vertex
			{
				std::stringstream ss(str);
				std::vector <std::string> record;

				char c;
				Type x, y, z;
				ss >> c >> x >> y >> z;
				vertices.push_back(x);
				vertices.push_back(y);
				vertices.push_back(z);
			}
			else if (str[1] == 'n')  // vertex
			{
				std::stringstream ss(str);
				std::vector <std::string> record;

				char c;
				Type x, y, z;
				ss >> c >> x >> y >> z;
				normals.push_back(x);
				normals.push_back(y);
				normals.push_back(z);
			}
		}
	}

	inFile.close();

	return true;
}


template<typename Type, int Rows>
static void export_obj_with_normals(const std::string& filename, const std::vector<Eigen::Matrix<Type, Rows, 1>>& vertices, const std::vector<Eigen::Matrix<Type, Rows, 1>>& normals)
{
	std::ofstream file;
	file.open(filename);
	for (const auto v : vertices)
		file << std::fixed << "v " << v.transpose() << std::endl;
	for (const auto n : normals)
		file << std::fixed << "vn " << n.transpose() << std::endl;
	file.close();
}

template<typename Type, int Rows>
static void export_obj_with_colors(const std::string& filename, const std::vector<Eigen::Matrix<Type, Rows, 1>>& vertices, const std::vector<Eigen::Vector3i>& rgb)
{
	std::ofstream file;
	file.open(filename);
	for (int i = 0; i < vertices.size(); ++i)
	{
		const auto& v = vertices[i];
		const auto& c = rgb[i];
		file << std::fixed << "v " << v.transpose() << '\t' << c.transpose() << std::endl;
	}
	file.close();
}


template<typename Type, int Rows>
static void export_obj_normals_as_colors(const std::string& filename, const std::vector<Eigen::Matrix<Type, Rows, 1>>& vertices, const std::vector<Eigen::Matrix<Type, Rows, 1>>& normals)
{
	std::ofstream file;
	file.open(filename);
	for (int i = 0; i < vertices.size(); ++i)
	{
		const auto& v = vertices[i];
		const auto& c = normals[i].abs() * 255;

		file << std::fixed << "v " << v.transpose() << '\t' << c.transpose() << std::endl;
	}
	file.close();
}


template<typename Type, int Rows>
static void export_normals_rgb(const std::string& filename, const std::vector<Eigen::Matrix<Type, Rows, 1>>& normals)
{
	std::ofstream file;
	file.open(filename);
	for (int i = 0; i < normals.size(); ++i)
	{
		const auto& c = normals[i].abs() * 255;

		file << std::fixed << "v " << c.transpose() << std::endl;
	}
	file.close();
}

template<typename Type>
void run_pca(
	const Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>& data_matrix,
	Eigen::Matrix<Type, 1, Eigen::Dynamic>& sorted_eigen_values,
	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>& sorted_eigen_vectors)
{
	//std::cout << "Data Matrix:\n" << data_matrix << std::endl << std::endl;

	// Compute a centered version of data matrix 
	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> centenered_data_matrix = data_matrix.rowwise() - data_matrix.colwise().mean();

	//std::cout << "Centered Data Matrix:\n" << centenered_data_matrix << std::endl << std::endl;

	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> covariance_matrix = (centenered_data_matrix.adjoint() * centenered_data_matrix) / (Type)(data_matrix.rows());

	//std::cout << "Covariance Matrix:\n" << covariance_matrix << std::endl << std::endl;

	// Use SelfAdjointEigenSolver to get eigen values and eigen vectors 
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>> eigen_solver(covariance_matrix);
	Eigen::Matrix<Type, 1, Eigen::Dynamic> eigen_values = eigen_solver.eigenvalues();
	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> eigen_vectors = eigen_solver.eigenvectors();

	// Stuff below is done to sort eigen values. This can be done in other ways too. 
	std::vector<std::pair<int, int>> eigen_value_index_vector;
	for (int i = 0; i < eigen_values.size(); ++i)
	{
		eigen_value_index_vector.push_back(std::make_pair(eigen_values[i], i));
	}
	std::sort(std::begin(eigen_value_index_vector), std::end(eigen_value_index_vector), std::greater<std::pair<int, int>>());

	sorted_eigen_values = Eigen::Matrix<Type, 1, Eigen::Dynamic>(eigen_values.cols());
	sorted_eigen_vectors = Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic>(eigen_vectors.rows(), eigen_vectors.cols());
	for (int i = 0; i < eigen_values.size(); ++i)
	{
		sorted_eigen_values[i] = eigen_values[eigen_value_index_vector[i].second]; //can also be eigen_value_index_vector[i].first
		sorted_eigen_vectors.col(i) = eigen_vectors.col(eigen_value_index_vector[i].second);
	}
	//std::cout << "Sorted Eigen Values:\n" << sorted_eigen_values << std::endl << std::endl;
	//std::cout << "Sorted Eigen Vectors(as columns):\n" << sorted_eigen_vectors << std::endl << std::endl;

	// Projection is W * X' 
	//Eigen::MatrixXd W = sorted_eigen_vectors.adjoint();
	//std::cout << "Y for 2-D projection:\n" << W.topRows(2) * centenered_data_matrix.adjoint() << std::endl;
}


template<typename Type>
void mesh_to_arrays(const aiMesh* mesh, Type*& vertex_array, Type*& normal_array)
{
	for (unsigned int i = 0; i<mesh->mNumFaces; i++)
	{
		const aiFace& face = mesh->mFaces[i];

		for (int j = 0; j<3; j++)
		{
			//aiVector3D uv = mesh->mTextureCoords[0][face.mIndices[j]];
			//memcpy(uv_array, &uv, sizeof(float) * 2);
			//uvArray += 2;

			aiVector3D normal = mesh->mNormals[face.mIndices[j]];
			memcpy(normal_array, &normal, sizeof(Type) * 3);
			normal_array += 3;

			aiVector3D pos = mesh->mVertices[face.mIndices[j]];
			memcpy(vertex_array, &pos, sizeof(Type) * 3);
			vertex_array += 3;
		}
	}
	normal_array -= mesh->mNumFaces * 3 * 3;
	vertex_array -= mesh->mNumFaces * 3 * 3;
}



template<typename Type>
void arrays_to_mesh(const Type* vertex_array, const Type* normal_array, aiMesh*& mesh)
{
	for (unsigned int i = 0; i<mesh->mNumFaces; i++)
	{
		const aiFace& face = mesh->mFaces[i];

		for (int j = 0; j<3; j++)
		{
			aiVector3D& normal = mesh->mNormals[face.mIndices[j]];
			memcpy(&normal, normal_array, sizeof(ai_real) * 3);
			normal_array += 3;

			aiVector3D& pos = mesh->mVertices[face.mIndices[j]];
			memcpy(&pos, vertex_array, sizeof(ai_real) * 3);
			vertex_array += 3;
		}
	}
	normal_array -= mesh->mNumFaces * 3 * 3;
	vertex_array -= mesh->mNumFaces * 3 * 3;
}

int using_assimp(const std::string& input_filename, const std::string& output_filename)
{
	typedef float Decimal;
	const int Dimension = 3;
	const int NumNeighbours = 5;

	//
	// Import file
	// 
	Assimp::Importer importer;
	const aiScene *scene = importer.ReadFile(input_filename, aiProcessPreset_TargetRealtime_Fast);//aiProcessPreset_TargetRealtime_Fast has the configs you'll need
	if (scene == nullptr)
	{
		std::cout << "Error: Could not read file: " << input_filename << std::endl;
		return EXIT_FAILURE;
	}

	aiMesh *mesh = scene->mMeshes[0]; //assuming you only want the first mesh
	std::cout
		<< "File        : " << input_filename << std::endl
		<< "Vertices    : " << mesh->mNumVertices << std::endl
		<< "Faces       : " << mesh->mNumFaces << std::endl
		<< "Has Normals : " << mesh->HasNormals() << std::endl;


	const size_t vertex_array_size = mesh->mNumFaces * 3 * 3;
	Decimal* vertex_array = new Decimal[vertex_array_size];
	Decimal* normal_array = new Decimal[vertex_array_size];

	mesh_to_arrays(mesh, vertex_array, normal_array);
	arrays_to_mesh(vertex_array, normal_array, mesh);

	Assimp::Exporter exporter;
	// [0 - dae], [1 - .x], [2 - .stp], [3 - .obj], [4 - .stl], [5 - .ply], 
	const aiExportFormatDesc* format = exporter.GetExportFormatDescription(3);
	aiReturn ret = exporter.Export(scene, format->id, output_filename, scene->mFlags);
	std::cout << "Error: " << exporter.GetErrorString()  << " ! " << std::endl;

	delete[] vertex_array;
	delete[] normal_array;

	return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
	typedef double Decimal;
	const int Dimension = 3;
	const int NumNeighbours = 5;

	std::string input_filename = "../../data/sphere.obj";
	std::string output_filename = "../../data/normal_estimated.obj";

	if (argc > 1)
		input_filename = argv[1];
	if (argc > 2)
		output_filename = argv[2];

	//return using_assimp(input_filename, output_filename);

	std::vector<Decimal> vertices, normals;
	if (!import_obj(input_filename, vertices, normals))
	{
		return EXIT_FAILURE;
	}

	
	// for each vertex find nearest neighbours
	const size_t NumInput = vertices.size() / Dimension;
	const size_t NumQuery = NumInput;

	flann::Matrix<Decimal> dataset(vertices.data(), NumInput, Dimension);
	flann::Matrix<Decimal> query(vertices.data(), NumQuery, Dimension);

	flann::Matrix<int> indices(new int[query.rows * NumNeighbours], query.rows, NumNeighbours);
	flann::Matrix<Decimal> dists(new Decimal[query.rows * NumNeighbours], query.rows, NumNeighbours);

	// construct an randomized kd-tree index using 4 kd-trees
	flann::Index<flann::L2<Decimal> > index(dataset, flann::KDTreeIndexParams(4));
	index.buildIndex();

	// do a knn search, using 128 checks
	//index.knnSearch(query, indices, dists, NumNeighbours, flann::SearchParams(128));
	index.knnSearch(query, indices, dists, NumNeighbours, flann::SearchParams(16));

	std::vector<Eigen::Matrix<Decimal, Dimension, 1>> normals_estimated;
	
	for (int i = 0; i < indices.rows; ++i)
	{
		const Decimal qx = query[i][0];
		const Decimal qy = query[i][1];
		const Decimal qz = query[i][2];

		Eigen::Matrix<Decimal, 1, Eigen::Dynamic> eigen_values;
		Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic> eigen_vectors;
		Eigen::Matrix<Decimal, NumNeighbours, Dimension> pca_data_matrix;	// indices.cols == NumNeighbours

		for (int j = 0; j < indices.cols; ++j)
		{
			// resultant points
			const int index = indices[i][j];
			const Decimal x = static_cast<Decimal>(dataset[index][0]);
			const Decimal y = static_cast<Decimal>(dataset[index][1]);
			const Decimal z = static_cast<Decimal>(dataset[index][2]);

			pca_data_matrix.row(j) << x, y, z;
		}


		run_pca<Decimal>(pca_data_matrix, eigen_values, eigen_vectors);

		normals_estimated.push_back( eigen_vectors.col(2) );
	}

	if (vertices.size() / Dimension != normals_estimated.size())
	{
		std::cerr << "Warning: Vertex and normal count don't match: "
			<< vertices.size() / Dimension << " != " << normals_estimated.size() << std::endl
			<< "Abort." << std::endl;
		return EXIT_FAILURE;
	}


	std::ofstream file;
	file.open(output_filename);
	for (int i = 0; i < normals_estimated.size(); ++i)
	{
		const auto& n = normals_estimated[i];
		file << std::fixed << "vn " << n.transpose() << std::endl;
	}
	file.close();

}