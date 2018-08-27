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

	Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic> covariance_matrix = (centenered_data_matrix.adjoint() * centenered_data_matrix) / (Type)(data_matrix.rows() - 1);

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


void flip_normals(aiMesh* mesh)
{

	for (int i = 0; i < mesh->mNumVertices; ++i)
	{
		mesh->mNormals[i] *= (-1);
		mesh->mVertices[i] *= 2;
	}

}


template<typename Type>
void copy_from_mesh(
	const aiMesh* mesh, 
	Type*& vertex_array,
	Type*& normal_array)
{
	Type* norm_ptr = normal_array;
	Type* vert_ptr = vertex_array;

	for (size_t i = 0; i < mesh->mNumVertices; ++i)
	{
		aiVector3D normal = mesh->mNormals[i];
		memcpy(normal_array, &normal, sizeof(Type) * 3);
		normal_array += 3;

		aiVector3D pos = mesh->mVertices[i];
		memcpy(vertex_array, &pos, sizeof(Type) * 3);
		vertex_array += 3;
	}
	vertex_array = vert_ptr;
	normal_array = norm_ptr;
}


template<typename Type>
void copy_normals_to_mesh(
	Type* normal_array, 
	aiMesh*& mesh)
{
	Type* norm_ptr = normal_array;

	for (size_t i = 0; i < mesh->mNumVertices; ++i)
	{
		aiVector3D& normal = mesh->mNormals[i];
		memcpy(&normal, normal_array, sizeof(Type) * 3);
		normal_array += 3;

	}
	normal_array = norm_ptr;
}



#define ASSIMP_DOUBLE_PRECISION
int main(int argc, char* argv[])
{
	std::cout
		<< std::endl
		<< "Usage            : ./<app.exe> <input_model> <output_format> <number_of_neighbours> <kd_tree_count> <knn_search_checks>" << std::endl
		<< "Default          : ./normal_estimation.exe ../../data/sphere.obj obj 16 4 128" << std::endl
		<< std::endl;

	typedef ai_real Decimal;
	const int Dimension = 3;
	const int NumNeighbours = (argc > 3) ? atoi(argv[3]) : 16;
	const int KdTreeCount = (argc > 4) ? atoi(argv[4]) : 4;
	const int KnnSearchChecks = (argc > 5) ? atoi(argv[5]) : 128;

	std::string input_filename = "../../data/sphere.obj";
	std::string output_format = "obj";

	if (argc > 1)
		input_filename = argv[1];
	if (argc > 2)
		output_format = argv[2];
	
	//
	// Composing output file name
	// 
	std::stringstream ss;
	ss << input_filename.substr(0, input_filename.size() - 4)
		<< '_' << NumNeighbours << '_' << KdTreeCount << '_' << KnnSearchChecks
		<< '.' << output_format;
	std::string output_filename = ss.str();


	//
	// Output info
	// 
	std::cout << std::fixed
		<< "Dimension        : " << Dimension << std::endl
		<< "NumNeighbours    : " << NumNeighbours << std::endl
		<< "KdTreeCount      : " << NumNeighbours << std::endl
		<< "KnnSearchChecks  : " << NumNeighbours << std::endl;


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


	//
	// Output info
	// 
	aiMesh *mesh = scene->mMeshes[0]; //assuming you only want the first mesh
	std::cout
		<< "File             : " << input_filename << std::endl
		<< "Vertices         : " << mesh->mNumVertices << std::endl
		<< "Faces            : " << mesh->mNumFaces << std::endl
		<< "Has Normals      : " << mesh->HasNormals() << std::endl;

	const size_t vertex_array_count = mesh->mNumVertices * Dimension;
	Decimal* vertex_array = new Decimal[vertex_array_count];
	Decimal* normal_array = new Decimal[vertex_array_count];
	copy_from_mesh(mesh, vertex_array, normal_array);
	

	// for each vertex find nearest neighbours
	const size_t NumInput = vertex_array_count / Dimension;
	const size_t NumQuery = NumInput;

	flann::Matrix<Decimal> dataset(vertex_array, NumInput, Dimension);
	flann::Matrix<Decimal> query(vertex_array, NumQuery, Dimension);

	flann::Matrix<int> indices(new int[query.rows * NumNeighbours], query.rows, NumNeighbours);
	flann::Matrix<Decimal> dists(new Decimal[query.rows * NumNeighbours], query.rows, NumNeighbours);

	// construct an randomized kd-tree index using 4 kd-trees
	flann::Index<flann::L2<Decimal> > index(dataset, flann::KDTreeIndexParams(KdTreeCount));
	index.buildIndex();

	// do a knn search, using 128 checks
	index.knnSearch(query, indices, dists, NumNeighbours, flann::SearchParams(KnnSearchChecks));	//flann::SearchParams(128));

	int n = 0;
	for (int i = 0; i < indices.rows; ++i)
	{
		const Decimal qx = query[i][0];
		const Decimal qy = query[i][1];
		const Decimal qz = query[i][2];

		Eigen::Matrix<Decimal, 1, Eigen::Dynamic> eigen_values;
		Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic> eigen_vectors;
		Eigen::Matrix<Decimal, Eigen::Dynamic, Eigen::Dynamic> pca_data_matrix(NumNeighbours, Dimension);	// indices.cols == NumNeighbours

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

		memcpy(&normal_array[i * Dimension], eigen_vectors.col(2).data(), sizeof(Decimal) * Dimension);
	}

	copy_normals_to_mesh(normal_array, mesh);

	Assimp::Exporter exporter;
	aiReturn ret = exporter.Export(scene, output_format, output_filename, scene->mFlags);
	if (ret == aiReturn_SUCCESS)
		std::cout << "Output File      : " << output_filename << std::endl;
	else
		std::cout << "Output File      : <ERROR> file not saved - " << output_filename << std::endl;



	delete[] vertex_array;
	delete[] normal_array;

	delete[] indices.ptr();
	delete[] dists.ptr();
}