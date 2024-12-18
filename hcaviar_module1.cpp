#include <iostream>
#include <vector>
#include <algorithm>
// #include <omp.h>
using namespace std;

long long comb(int n, int k) {

    if (k > n) return 0;
    if (k == 0 || k == n) return 1;

    // Create a 1D DP array
    std::vector<long long> dp(k + 1, 0);
    dp[0] = 1; // comb(i, 0) = 1

    // Fill DP array using Pascal's Triangle property
    for (int i = 1; i <= n; ++i) {
        for (int j = std::min(i, k); j > 0; --j) {
            dp[j] += dp[j - 1];
        }
    }

    return dp[k];
}

vector<vector<bool>> generate_combinations_batch(int M, int P, int batch_size, vector<bool> v_start, bool f_discard_1st=false) {

    vector<vector<bool>> result;
    result.reserve(batch_size);

    // cout << "start vector:" << endl;
    // for (auto v : v_start) {
    //     cout << v << " ";
    // }cout << endl;


    if (f_discard_1st) {
        // discard the 1st item, which is redundant from the 2nd batch.
        std::prev_permutation(v_start.begin(), v_start.end());
    }

    int i = 0;
    do {
        result.push_back(v_start);
        i++;
    } while (i < batch_size && prev_permutation(v_start.begin(), v_start.end()));

    return result;

}



int main() {

    int M = 50, P = 2, batch_size = 32;

    int total_combinations = comb(M, P);
    int total_batches = (total_combinations + batch_size - 1) / batch_size; // 올림 처리 (걍 ceil function 구현한 거)
    int start_index = 0;

    printf("%d\n", total_combinations);

    // initializaiton of configuration vectors.
    vector<bool> temp (M, 0);
    fill(temp.begin(), temp.begin() + P, 1);

    vector<vector<bool>> configurations_batch;
    configurations_batch.push_back(temp);


    for (int batch_idx = 0; batch_idx < total_batches; batch_idx++) {

        configurations_batch =
            generate_combinations_batch(M, P, batch_size, configurations_batch.back(), (batch_idx > 0));

        // #pragma omp parallel for
        for (int i = 0; i < configurations_batch.size(); i++) {

            // cout << "\t===[" << batch_idx*batch_size + i << "/" << i << "]" << endl;
            cout << "\t===[" << batch_idx*batch_size + i << "]" << endl;

            for (int bit : configurations_batch[i]) {
                std::cout << bit << " ";
            } std::cout << std::endl;

        }
    }

    return 0;

}