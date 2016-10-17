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
void copy_vertices_from_mesh(
	const aiMesh* mesh, 
	Type*& vertex_array)
{
	for (unsigned int i = 0; i<mesh->mNumFaces; i++)
	{
		const aiFace& face = mesh->mFaces[i];

		for (int j = 0; j<3; j++)
		{
			//aiVector3D uv = mesh->mTextureCoords[0][face.mIndices[j]];
			//memcpy(uv_array, &uv, sizeof(float) * 2);
			//uvArray += 2;

			//aiVector3D normal = mesh->mNormals[face.mIndices[j]];
			//memcpy(normal_array, &normal, sizeof(Type) * 3);
			//normal_array += 3;

			aiVector3D pos = mesh->mVertices[face.mIndices[j]];
			memcpy(vertex_array, &pos, sizeof(Type) * 3);
			vertex_array += 3;
		}
	}
	vertex_array -= mesh->mNumFaces * 3 * 3;
}


template<typename Type>
void copy_normals_to_mesh(
	const Type* normal_array, 
	aiMesh*& mesh)
{
	for (unsigned int i = 0; i<mesh->mNumFaces; i++)
	{
		const aiFace& face = mesh->mFaces[i];

		for (int j = 0; j<3; j++)
		{
			aiVector3D& normal = mesh->mNormals[face.mIndices[j]];
			memcpy(&normal, normal_array, sizeof(ai_real) * 3);
			normal_array += 3;
		}
	}
	normal_array -= mesh->mNumFaces * 3 * 3;
}




int main(int argc, char* argv[])
{
	std::cout
		<< std::endl
		<< "Usage  : ./normal_estimation.exe input.obj output.obj" << std::endl
		<< "Default: ./normal_estimation.exe ../../data/sphere.obj ../../data/normal_estimated.obj" << std::endl
		<< std::endl;

	typedef double Decimal;
	const int Dimension = 3;
	const int NumNeighbours = 5;

	std::string input_filename = "../../data/sphere.obj";
	std::string output_filename = "../../data/normal_estimated.obj";

	if (argc > 1)
		input_filename = argv[1];
	if (argc > 2)
		output_filename = argv[2];
	

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
	copy_vertices_from_mesh(mesh, vertex_array);
	
	// for each vertex find nearest neighbours
	const size_t NumInput = vertex_array_size / Dimension;
	const size_t NumQuery = NumInput;

	flann::Matrix<Decimal> dataset(vertex_array, NumInput, Dimension);
	flann::Matrix<Decimal> query(vertex_array, NumQuery, Dimension);

	flann::Matrix<int> indices(new int[query.rows * NumNeighbours], query.rows, NumNeighbours);
	flann::Matrix<Decimal> dists(new Decimal[query.rows * NumNeighbours], query.rows, NumNeighbours);

	// construct an randomized kd-tree index using 4 kd-trees
	flann::Index<flann::L2<Decimal> > index(dataset, flann::KDTreeIndexParams(4));
	index.buildIndex();

	// do a knn search, using 128 checks
	index.knnSearch(query, indices, dists, NumNeighbours, flann::SearchParams(32));	//flann::SearchParams(128));

	int n = 0;
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

		normal_array[n++] = eigen_vectors.col(2)[0];
		normal_array[n++] = eigen_vectors.col(2)[1];
		normal_array[n++] = eigen_vectors.col(2)[2];
	}

	copy_normals_to_mesh(normal_array, mesh);

	Assimp::Exporter exporter;
	// [0 - dae], [1 - .x], [2 - .stp], [3 - .obj], [4 - .stl], [5 - .ply], 
	const aiExportFormatDesc* format = exporter.GetExportFormatDescription(3);
	aiReturn ret = exporter.Export(scene, format->id, output_filename, scene->mFlags);

	delete[] vertex_array;
	delete[] normal_array;

	delete[] indices.ptr();
	delete[] dists.ptr();
}