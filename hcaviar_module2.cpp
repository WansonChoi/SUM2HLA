#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <set>

#include <armadillo>

using namespace std;

struct LDMatrix {

	// std::unordered_map<std::string, int> col_idx_HLA; // 헤더 이름 -> 인덱스 매핑

	std::map<int, std::string> col_idx_HLA;
	std::map<int, std::string> col_idx_SNP;
	int M = -1;
	arma::mat matrix;                                // LD matrix 데이터

	// 파일에서 데이터 로드
	bool load(const std::string& filename) {

		std::ifstream file(filename);

		if (!file.is_open()) {
			std::cerr << "Error: Unable to open file " << filename << std::endl;
			return false;
		}

		std::string line;

		// 첫 번째 줄: 헤더 읽기
		if (std::getline(file, line)) {

			std::istringstream iss(line);
			std::string header;

			for (M = 0; iss >> header; ++M) {
				if (header.find("HLA") == 0) { // "HLA"로 시작하는지 확인
					col_idx_HLA[M] = header;
				}
				else {
					col_idx_SNP[M] = header;
				}

			}
		} else {
			std::cerr << "Error: File is empty or missing header line." << std::endl;
			return false;
		}

		// 나머지 줄: 행렬 데이터 읽기
		matrix = arma::mat(M, M);  // Armadillo 행렬 초기화

		for (int row = 0; std::getline(file, line) && row < M; ++row) {

			std::istringstream iss(line);

			for (int col = 0; col < M; ++col) {

				double value;

				if (iss >> value) {
					matrix(row, col) = value;
				} else {
					std::cerr << "Error: Insufficient data in row " << row << std::endl;
					return false;
				}
			}
		}

		return true;
	}
};


arma::mat subset_matrix(arma::mat& _mat, std::map<int, std::string>& col_idx_SNP) {

	arma::uvec indices_to_keep;
	for (const auto& [key, index] : col_idx_SNP) {

		indices_to_keep.insert_rows(indices_to_keep.n_rows, arma::uvec({(arma::uword)key}));
	}

	// 행렬 subset
	return _mat(indices_to_keep, indices_to_keep);
}


arma::vec get_ncp_mean_vector(arma::mat& LD_mat, const std::vector<bool>& _a_configure, double _ncp = 5.2) {

	// (1) `a_configure`를 Armadillo 벡터로 변환
	arma::uvec a_configure_uvec(_a_configure.size());
	for (size_t i = 0; i < _a_configure.size(); ++i) {
		a_configure_uvec(i) = _a_configure[i] ? 1 : 0; // bool을 1 또는 0으로 변환
	}

	// (2) Dot product: (13x13) * (13x1) -> (13x1)
	arma::vec ncp_vector = _ncp * (LD_mat * arma::conv_to<arma::vec>::from(a_configure_uvec));

	// (3) `a_configure`에서 1인 위치는 값을 `_ncp`로 설정
	for (size_t i = 0; i < _a_configure.size(); ++i) {
		if (_a_configure[i]) {
			ncp_vector(i) = _ncp;
		}
	}

	return ncp_vector; // (13x1) 벡터 반환
}



// Subset function
arma::vec subset_vector(const arma::vec& ncp_mean_vector, const std::map<int, std::string>& col_idx_SNP) {
	// SNP 인덱스 추출
	arma::uvec indices_to_keep(col_idx_SNP.size());
	size_t idx = 0;
	for (const auto& [key, value] : col_idx_SNP) {
		indices_to_keep(idx++) = key;
	}

	// Subset 벡터 생성
	return ncp_mean_vector(indices_to_keep);
}



struct GWAS_Z_observed {

	// Observed z-scores 저장
	std::unordered_map<std::string, double> z_scores;
	std::vector<double> z_scores_sorted;


	// (1) Load 함수: SNP z-scores 데이터 로드
	bool load(const std::string& filename) {
		std::ifstream file(filename);
		if (!file.is_open()) {
			std::cerr << "Error: Unable to open file " << filename << std::endl;
			return false;
		}

		std::string line, header, snp_label;
		double z_score;

		// 첫 번째 줄: 헤더 확인
		if (!std::getline(file, line)) {
			std::cerr << "Error: File is empty or missing header line." << std::endl;
			return false;
		}

		// 이후 줄: SNP와 z-score 저장
		while (std::getline(file, line)) {
			std::istringstream iss(line);
			if (!(iss >> snp_label >> z_score)) {
				std::cerr << "Error: Invalid line format: " << line << std::endl;
				return false;
			}
			z_scores[snp_label] = z_score; // SNP-z-score 매핑 저장
		}

		return true;
	}

	// (2) LD matrix의 col_idx_SNP 순서에 따라 z-scores 정렬
	void sort_z_scores(const std::map<int, std::string>& col_idx_SNP) {

		// (a) SNP set 비교
		std::set<std::string> observed_snps, ld_snps;
		for (const auto& [snp_label, z_score] : z_scores) {
			observed_snps.insert(snp_label);
		}
		for (const auto& [idx, snp_label] : col_idx_SNP) {
			ld_snps.insert(snp_label);
		}

		if (observed_snps != ld_snps) {
			throw std::runtime_error("Error: SNP sets in observed z-scores and LD matrix do not match.");
		}

		// (b) LD matrix의 col_idx_SNP 순서에 따라 정렬
		for (const auto& [idx, snp_label] : col_idx_SNP) {
			z_scores_sorted.push_back(z_scores[snp_label]);
		}

	}
};



// Term 1: -0.5 * M * log(2 * pi)
double calc_term1(int M) {
	const double pi = arma::datum::pi; // Armadillo의 π 상수
	return -0.5 * M * std::log(2 * pi);
}

// Term 2: -0.5 * log(det(corr_matrix)) using log of eigenvalues
double calc_term2(const arma::mat& corr_matrix) {
	// 고윳값 계산 (대칭행렬 가정)
	arma::vec eigenvalues = arma::eig_sym(corr_matrix); // 대칭행렬의 고윳값
	// log(det(corr_matrix)) = sum(log(eigenvalues))
	double log_det = arma::sum(arma::log(eigenvalues));
	return -0.5 * log_det;
}

// Log-likelihood 계산
double calc_log_likelihood_mvn(
	const arma::vec& obs_zscores,         // Observed z-scores
	const arma::vec& mean_vector,        // Mean vector
	const arma::mat& inv_corr_matrix,    // Inverse of correlation matrix
	double term1,                        // Precomputed Term 1
	double term2                         // Precomputed Term 2
) {
	// diff = z_scores - mean_vector
	arma::vec diff = obs_zscores - mean_vector;

	// Term 3: -0.5 * (z_scores - mean_vector)^T @ inv(corr_matrix) @ (z_scores - mean_vector)
	double mahalanobis_dist = arma::as_scalar(diff.t() * inv_corr_matrix * diff); // Mahalanobis distance
	double term3 = -0.5 * mahalanobis_dist;

	// Total log-likelihood
	return term1 + term2 + term3;
}


// 검산용 (거의 똑같아서 의미 없긴 함;;
double calc_log_pdf_mvn(
	const arma::vec& obs_zscores,        // Observed z-scores
	const arma::vec& mean_vector,       // Mean vector
	const arma::mat& ld_matrix          // LD matrix (not inverse)
) {
	const double pi = arma::datum::pi;
	int k = ld_matrix.n_rows; // Dimension (M)

	// (1) -0.5 * k * log(2 * pi)
	double term1 = -0.5 * k * std::log(2 * pi);

	// (2) Determinant and inverse of LD matrix
	arma::mat inv_ld_matrix;
	double det_ld_matrix;
	bool success = arma::inv(inv_ld_matrix, ld_matrix); // Inverse
	if (!success) {
		throw std::runtime_error("Error: LD matrix is singular and cannot be inverted.");
	}
	det_ld_matrix = arma::det(ld_matrix);
	if (det_ld_matrix <= 0) {
		throw std::runtime_error("Error: Determinant of LD matrix is non-positive.");
	}
	double term2 = -0.5 * std::log(det_ld_matrix);

	// (3) Mahalanobis distance: -0.5 * (z_scores - mean_vector)^T @ inv(ld_matrix) @ (z_scores - mean_vector)
	arma::vec diff = obs_zscores - mean_vector;
	double mahalanobis_dist = arma::as_scalar(diff.t() * inv_ld_matrix * diff);
	double term3 = -0.5 * mahalanobis_dist;

	// Total log-pdf
	return term1 + term2 + term3;
}


int main() {

	vector<bool> a_configure(13, false);
	a_configure[0] = true;
	a_configure[1] = true;

	for (int i = 0; i < 10; i++) {
		cout << a_configure[i] << " ";
	}cout << endl;



	// === [0] 파일 로드

	LDMatrix ld_matrix;

	if (!ld_matrix.load("./sample_data/C++_example.LD.M10+3.txt")) {
		std::cerr << "Failed to load the LD Matrix." << std::endl;
	}

	std::cout << "LD Matrix successfully loaded!" << std::endl;

	// 예: 행렬 출력
	ld_matrix.matrix.print("Loaded Matrix:"); // armadillo 자체 함수가 있나 봄.

	// HLA marker들의 index들 확인
	std::cout << "col_inx_HLA contains " << ld_matrix.col_idx_HLA.size() << " items:" << std::endl;

	for (const auto& pair : ld_matrix.col_idx_HLA) {
		std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
	}
	for (const auto& pair : ld_matrix.col_idx_SNP) {
		std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
	}


	GWAS_Z_observed z_observed;

	if (!z_observed.load("sample_data/C++_example.Z.M10.txt")) {
		std::cerr << "Failed to load observed z-scores." << std::endl;
	}

	try {

		// z-scores 정렬
		z_observed.sort_z_scores(ld_matrix.col_idx_SNP);

		// 정렬된 z-scores 출력
		std::cout << "Sorted z-scores:" << std::endl;
		for (double z : z_observed.z_scores_sorted) {
			std::cout << z << std::endl;
		}

	} catch (const std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
	}



	// === [1] ncp mean vector 계산
	arma::vec ncp_mean_vector = get_ncp_mean_vector(ld_matrix.matrix, a_configure);
	ncp_mean_vector.print("NCP mean vector:");

	arma::vec subset_ncp_mean_vector = subset_vector(ncp_mean_vector, ld_matrix.col_idx_SNP);
	subset_ncp_mean_vector.print("Subset NCP mean vector:");


	// === [2] HLA markers들 제외: (1) ncp mean vector, (2) LD matrix
	ld_matrix.matrix = subset_matrix(ld_matrix.matrix, ld_matrix.col_idx_SNP);
	ld_matrix.matrix.print("Loaded Matrix (subsetted):"); // armadillo 자체 함수가 있나 봄.



	// 여기서 부터는 test상 그냥 각각 새로운 matrix만드는 걸로 일단 예시 굴리기.
	// 나중에는 계속 같은 변수에 re-assign해야함.

	// likelihood 계산할대는 HLA marekers 제외된 LD matrix를 써야 함: (1) eigen values, (2) inverse matrix.


	double log_likelihood_val = calc_log_pdf_mvn(
		arma::vec(z_observed.z_scores_sorted), subset_ncp_mean_vector, ld_matrix.matrix);
	cout << "Validation of Log-likelihood: " << log_likelihood_val << endl;
	printf("%lf\n", log_likelihood_val);



	// === [3] term1, term2 (eigenvalues) 계산.
	double term1 = calc_term1(ld_matrix.M);
	double term2 = calc_term2(ld_matrix.matrix); // 얘가 inverse구하기 전에 계산해야 함.


	// === [4] log-likelihood

	// === [4-1] inverse matrix 계산

	bool success = arma::inv(ld_matrix.matrix, ld_matrix.matrix); // `ld_matrix.matrix`에 reassign
	cout << "Getting inverse matrix succeeded? " << success << endl;
	ld_matrix.matrix.print("Inverse Matrix:");

	arma::vec obs_zscores = arma::vec(z_observed.z_scores_sorted);


	// === [4-2] Compute log-likelihood (:= term1 + term2 + term3)
	double log_likelihood = calc_log_likelihood_mvn(
		obs_zscores, subset_ncp_mean_vector, ld_matrix.matrix, term1, term2
	);

	cout << "Log-likelihood: " << log_likelihood << endl;
	printf("%lf\n", log_likelihood);



	return 0;

};