#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <Eigen/Dense>
#include <flann/flann.hpp>

#include "tinyply.h"
#include "timer.h"
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;




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


bool load_ply(
	const std::string& input_filename, 
	std::vector<float>& verts,
	std::vector<float>& norms, 
	std::vector<float>& uvCoords,
	std::vector<uint32_t>& faces,
	std::vector<uint8_t>& colors)
{
	try
	{
		std::cout << "Loading          : " << input_filename << std::endl;

		std::ifstream ss_temp(input_filename, std::ios::binary);
		tinyply::PlyFile file_template(ss_temp);

		uint64_t vertexCount = file_template.request_properties_from_element("vertex", { "x", "y", "z" }, verts);
		uint64_t normalCount = file_template.request_properties_from_element("vertex", { "nx", "ny", "nz" }, norms);
		uint64_t colorCount = file_template.request_properties_from_element("vertex", { "red", "green", "blue", "alpha" }, colors);
		uint64_t faceCount = file_template.request_properties_from_element("face", { "vertex_indices" }, faces, 3);
		uint64_t faceTexcoordCount = file_template.request_properties_from_element("face", { "texcoord" }, uvCoords, 6);

		if (vertexCount != (verts.size() / 3))
		{
			std::cout << "Error: Only triangle mesh is supported. Abort" << std::endl;
			return false;
		}

		file_template.read(ss_temp);

		std::cout
			<< "Vertices         : " << (!verts.empty() ? verts.size() / 3 : 0) << std::endl
			<< "Faces            : " << faces.size() << std::endl
			<< "Normals          : " << (!norms.empty() ? norms.size() / 3 : 0) << std::endl
			<< "UV Coords        : " << (!uvCoords.empty() ? uvCoords.size() / 2 : 0) << std::endl;
	}
	catch (const std::exception & e)
	{
		std::cerr << "Error: Could not load " << input_filename << ". " << e.what() << std::endl;
		return false;
	}
	return true;
}

bool save_ply(
	const std::string& output_filename,
	std::vector<float>& verts,
	std::vector<float>& norms,
	std::vector<float>& uvCoords,
	std::vector<uint32_t>& faces,
	std::vector<uint8_t>& colors)
{
	try
	{
		std::cout << "Saving           : " << output_filename << std::endl;

		std::filebuf fb;
		fb.open(output_filename, std::ios::out | std::ios::binary);
		std::ostream outputStream(&fb);

		tinyply::PlyFile ply_out_file;

		ply_out_file.add_properties_to_element("vertex", { "x", "y", "z" }, verts);
		if (!norms.empty())
			ply_out_file.add_properties_to_element("vertex", { "nx", "ny", "nz" }, norms);
		if (!colors.empty())
			ply_out_file.add_properties_to_element("vertex", { "red", "green", "blue", "alpha" }, colors);
		if (!faces.empty())
			ply_out_file.add_properties_to_element("face", { "vertex_indices" }, faces, 3, tinyply::PlyProperty::Type::UINT8);
		if (!uvCoords.empty())
			ply_out_file.add_properties_to_element("face", { "texcoord" }, uvCoords, 6, tinyply::PlyProperty::Type::UINT8);

		ply_out_file.write(outputStream, true);

		fb.close();

		std::cout
			<< "Vertices         : " << (!verts.empty() ? verts.size() / 3 : 0) << std::endl
			<< "Faces            : " << faces.size() << std::endl
			<< "Normals          : " << (!norms.empty() ? norms.size() / 3 : 0) << std::endl
			<< "UV Coords        : " << (!uvCoords.empty() ? uvCoords.size() / 2 : 0) << std::endl;
	}
	catch (const std::exception & e)
	{
		std::cerr << "Error: Could not save " << output_filename << ". " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
}


int main(int argc, char* argv[])
{
	if (argc < 4)
	{
		std::cout
			<< std::endl
			<< "Usage            : ./<app.exe> <input_ply> <output_ply> <number_of_neighbours> <kd_tree_count> <knn_search_checks>" << std::endl
			<< "Default          : ./normal_estimation.exe ../../data/sphere.ply ply 16 4 128" << std::endl
			<< std::endl;
		return EXIT_FAILURE;
	}

	typedef float Decimal;
	const int Dimension = 3;

	const std::string input_filename = (argc > 1) ? argv[1] : "../../data/teddy.obj";
	const std::string output_filename = (argc > 2) ? argv[2] : "output.obj";
	const std::string extension = fs::path(output_filename).extension().string();
	const std::string output_format = extension.substr(1, extension.length() - 1);
	const int NumNeighbours = (argc > 3) ? atoi(argv[3]) : 16;
	const int KdTreeCount = (argc > 4) ? atoi(argv[4]) : 4;
	const int KnnSearchChecks = (argc > 5) ? atoi(argv[5]) : 128;

	if (output_format != "ply")
	{
		std::cout << "File format not suported. Abort" << std::endl;
		return 1;
	}
	

	//
	// Output info
	// 
	std::cout << std::fixed
		<< "Input            : " << input_filename << std::endl
		<< "Output           : " << output_filename << std::endl
		<< "Dimension        : " << Dimension << std::endl
		<< "NumNeighbours    : " << NumNeighbours << std::endl
		<< "KdTreeCount      : " << KdTreeCount << std::endl
		<< "KnnSearchChecks  : " << KnnSearchChecks << std::endl;

	std::vector<Decimal> verts;
	std::vector<Decimal> norms, uvCoords;
	std::vector<uint32_t> faces;
	std::vector<uint8_t> colors;
	
	timer tm_load;
	if (!load_ply(input_filename, verts, norms, uvCoords, faces, colors))
	{
		std::cout << "Error: Could not load ply file." << std::endl;
		return EXIT_FAILURE;
	}
	tm_load.stop();

	std::cout << "Building kd-tree... " << std::endl;

	timer tm_kdtree;
	// for each vertex find nearest neighbours
	const size_t NumInput = verts.size() / Dimension;
	const size_t NumQuery = NumInput;

	flann::Matrix<Decimal> dataset(verts.data(), NumInput, Dimension);
	flann::Matrix<Decimal> query(verts.data(), NumQuery, Dimension);

	flann::Matrix<int> indices(new int[query.rows * NumNeighbours], query.rows, NumNeighbours);
	flann::Matrix<Decimal> dists(new Decimal[query.rows * NumNeighbours], query.rows, NumNeighbours);

	// construct an randomized kd-tree index using 4 kd-trees
	flann::Index<flann::L2<Decimal> > index(dataset, flann::KDTreeIndexParams(KdTreeCount));
	index.buildIndex();

	// do a knn search, using 128 checks
	index.knnSearch(query, indices, dists, NumNeighbours, flann::SearchParams(KnnSearchChecks));	//flann::SearchParams(128));
	tm_kdtree.stop();

	timer tm_pca;
	int n = 0;
	for (int i = 0; i < indices.rows; ++i)
	{
		std::cout << "Computing row    " << i << '\r';

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

		memcpy(&norms[i * Dimension], eigen_vectors.col(2).data(), sizeof(Decimal) * Dimension);
	}
	std::cout << std::endl;
	tm_pca.stop();

	delete[] indices.ptr();
	delete[] dists.ptr();

	//
	// Normalizing normals
	//
	timer tm_nor;
	for (auto i = 0; i < norms.size(); i += 3)
	{
		auto length = std::sqrt((norms[i] * norms[i]) + (norms[i + 1] * norms[i + 1]) + (norms[i + 2] * norms[i + 2]));
		norms[i] /= length; 
		norms[i+1] /= length; 
		norms[i+2] /= length;
	}
	tm_nor.stop();

	timer tm_save;
	if (!save_ply(output_filename, verts, norms, uvCoords, faces, colors))
	{
		std::cout << "Error: Could not save " << output_filename << std::endl;
		return EXIT_FAILURE;
	}
	tm_save.stop();

	std::cout << std::fixed
		<< "[Times in seconds]  \n"
		<< "Loading          : " << tm_load.diff_sec() << '\n'
		<< "Building Kd-Tree : " << tm_kdtree.diff_sec() << '\n'
		<< "Computing Normals: " << tm_pca.diff_sec() << '\n'
		<< "Normalizing Vecs : " << tm_nor.diff_sec() << '\n'
		<< "Saving           : " << tm_save.diff_sec() << '\n'
		<< std::endl;

	return EXIT_SUCCESS;
}