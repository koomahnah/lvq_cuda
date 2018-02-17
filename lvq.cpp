#include "cuda.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <map>
#include <set>

using namespace std;
namespace fs = boost::filesystem;

#define CHECK_ERROR(X...)            \
    do {                            \
        CUresult res;               \
        res = X;          \
        if (res != CUDA_SUCCESS){   \
            cout << "error at: " << endl;   \
            cout << (#X) << endl;             \
            exit(1);                \
        }                           \
    } while(0);


#define THREADS_PER_BLOCK 32

string convert_word(string in) {
    string out;
    transform(in.begin(), in.end(), in.begin(), ::tolower);
    remove_copy_if(in.begin(), in.end(), back_inserter(out), ptr_fun<int,int>(&ispunct));
    out.erase(std::remove_if(out.begin(), out.end(),
                             [](char ch){ return !::isalpha(ch); }), out.end());
    return out;
}

vector<bool> text_to_vec(fs::path text, map<string,int> dict) {
    std::ifstream file;
    string word;
    vector<bool> vec(dict.size(), 0);

    cout << "opening " << text.c_str() << endl;
    file.open(text.c_str());
    while (file >> word) {
        string out = convert_word(word);

        cout << "on word " << out << endl;
        if (out.length() < 2)
            continue;

        if (dict.find(out) != dict.end()) {
            int word_index = dict[out];
            assert(word_index < vec.size());
            vec[word_index] =  true;
        }
    }
    file.close();

    return vec;
}

map <string,int> create_dict(fs::path target_dir) {
    std::ifstream file;
    fs::directory_iterator it(target_dir), eod;
    int word_index = 0;
    map<string, int> word_dict;

    BOOST_FOREACH(fs::path const &p, make_pair(it, eod)) {
        assert(fs::is_directory(p));
        fs::recursive_directory_iterator it2(p), eod;
        cout << "on directory " << p << endl;
        int count = 0;
        BOOST_FOREACH(fs::path const &subp, make_pair(it2, eod)) {
            if (count++ > 100)
                break;
            string word;
            cout << "on file " << subp << endl;

            file.open(subp.c_str());
            if (!fs::is_regular_file(subp))
                cout << "skipping " << subp << endl;

            while (file >> word) {
                string out = convert_word(word);

                if (out.length() < 2)
                    continue;

                if (word_dict.find(out) == word_dict.end()) {
                    word_dict[out] = word_index++;
                    cout << word_index - 1 << " " << out << endl;
                }
            }
            cout << endl;
            file.close();
        }
    }
    cout << word_dict.size() << endl;
    return word_dict;
}

void build_text_array(fs::path target_dir, map<string,int> dict, int input_dim,
                      int **text_array_out, int **text_class_out, int *text_cnt_out) {
    std::ifstream file;
    fs::directory_iterator it(target_dir), eod;
    int word_index = 0;
    int text_cnt = std::count_if(
                   fs::recursive_directory_iterator(target_dir),
                   fs::recursive_directory_iterator(),
                   static_cast<bool(*)(const fs::path&)>(fs::is_regular_file) );

    cout << text_cnt << " files in directory" << endl;

    int *text_array = new int[text_cnt * input_dim];
    int *text_class = new int[text_cnt];

    for (int i = 0; i < text_cnt * input_dim; i++)
        text_array[i] = 0;

    int text_id = 0;
    int class_id = 0;
    BOOST_FOREACH(fs::path const &p, make_pair(it, eod)) {
        assert(fs::is_directory(p));
        fs::recursive_directory_iterator it2(p), eod;
        cout << "on directory " << p << endl;
        int count = 0;
        BOOST_FOREACH(fs::path const &subp, make_pair(it2, eod)) {
            if (count++ > 100)
                break;
            string word;
            cout << "on file " << subp << endl;

            file.open(subp.c_str());
            if (!fs::is_regular_file(subp))
                cout << "skipping " << subp << endl;

            while (file >> word) {
                string out = convert_word(word);

                if (out.length() < 2)
                    continue;

                if (dict.find(out) != dict.end()) {
                    int word_index = dict[out];
                    assert(word_index < input_dim);
                    text_array[text_id * input_dim + word_index] =  1;
                }
            }
            text_class[text_id] = class_id;
            text_id++;
            cout << endl;
            file.close();
        }
        class_id++;
    }
    *text_array_out = text_array;
    *text_class_out = text_class;
    *text_cnt_out = text_cnt;
}

int compete(CUfunction distance, int input_dim,
            CUdeviceptr neuron_weight_d, int neuron_count,
            CUdeviceptr text_array_d, int text_index,
            CUdeviceptr neuron_dist_d)
{
    static int *neuron_dist = new int[neuron_count];

    void* args[] = {&input_dim, &neuron_weight_d, &text_array_d, &text_index,
                     &neuron_dist_d};
    CHECK_ERROR(cuLaunchKernel(distance, neuron_count,
            1, 1, THREADS_PER_BLOCK, 1, 1, 0, 0, args, 0));
    CHECK_ERROR(cuCtxSynchronize());

    cuMemcpyDtoH(neuron_dist, neuron_dist_d, sizeof(int) * neuron_count);

    float min_dist = numeric_limits<float>::max();
    int min_index = -1;
    for (int i = 0; i < neuron_count; i++) {
        if (neuron_dist[i] < min_dist) {
            min_index = i;
            min_dist = neuron_dist[i];
        }
    }
    assert(min_index != -1);

    return min_index;
}

int main() {
    cuInit(0);

    CUdevice device;
    CUcontext context;
    CUmodule cuModule = (CUmodule)0;

    CHECK_ERROR(cuDeviceGet(&device, 0));
    CHECK_ERROR(cuCtxCreate(&context, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, device));
    CHECK_ERROR(cuModuleLoad(&cuModule, "lvq.ptx"));

    map<string,int> dict = create_dict(fs::path("./texts"));
    int input_dim = dict.size();
    int neuron_count = 4;
    int text_count;
    int output_dim = 2; // number of classes

    cout << "Created dict. Building text array..." << endl;

    int *text_array, *text_class;

    build_text_array(fs::path("./texts"), dict, input_dim, &text_array,
            &text_class, &text_count);

    cout << "Text count is " << text_count << ", input dim " << input_dim << endl;

    for (int i = 0; i < text_count * input_dim; i++) {
        cout << text_array[i] << " ";
        if (i % input_dim == input_dim - 1)
            cout << endl;
    }
    cout << endl;

    CUdeviceptr neuron_class_d;
    CUdeviceptr neuron_weight_d;
    CUdeviceptr neuron_bias_d;
    CUdeviceptr text_array_d;
    CUdeviceptr neuron_dist_d;

    CUfunction init;
    CUfunction distance;

    CHECK_ERROR(cuMemAlloc(&neuron_class_d, neuron_count * sizeof(int)));
    CHECK_ERROR(cuMemAlloc(&neuron_weight_d, neuron_count * input_dim * sizeof(float)));
    CHECK_ERROR(cuMemAlloc(&neuron_bias_d, neuron_count * sizeof(int)));
    CHECK_ERROR(cuMemAlloc(&neuron_dist_d, neuron_count * sizeof(float)));
    CHECK_ERROR(cuMemAlloc(&text_array_d, text_count * input_dim * sizeof(int)));

    CHECK_ERROR(cuMemcpyHtoD(text_array_d, text_array, text_count * input_dim * sizeof(int)));

    CHECK_ERROR(cuModuleGetFunction(&init, cuModule, "init"));
    CHECK_ERROR(cuModuleGetFunction(&distance, cuModule, "distance"));

    void* args[] = {&input_dim, &output_dim, &neuron_count, &neuron_class_d, &neuron_weight_d,
                    &neuron_bias_d};
    CHECK_ERROR(cuLaunchKernel(init, neuron_count/2,
            1, 1, 2, 1, 1, 0, 0, args, 0));
    CHECK_ERROR(cuCtxSynchronize());

    int winner = compete(distance, input_dim, neuron_weight_d, neuron_count,
            text_array_d, 0, neuron_dist_d);
    cuCtxDestroy(context);

    return 0;
}

