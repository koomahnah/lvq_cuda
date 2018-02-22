#include "cuda.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
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

struct {
    int input_dim;
    int output_dim;
    int neuron_count;
    int text_count;

    CUfunction init;
    CUfunction distance;
    CUfunction attract;

    int *text_array;
    int *text_class;
    double *neuron_dist;
    int *neuron_bias;
    int *neuron_class;

    CUdeviceptr neuron_weight_d;
    CUdeviceptr neuron_dist_d;
    CUdeviceptr text_array_d;
} l;

string convert_word(string in) {
    string out;
    transform(in.begin(), in.end(), in.begin(), ::tolower);
    remove_copy_if(in.begin(), in.end(), back_inserter(out), ptr_fun<int,int>(&ispunct));
    out.erase(std::remove_if(out.begin(), out.end(),
                             [](char ch){ return !::isalpha(ch); }), out.end());
    return out;
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
            string word;
//            cout << "on file " << subp << endl;

            file.open(subp.c_str());
            if (!fs::is_regular_file(subp))
                cout << "skipping " << subp << endl;

            while (file >> word) {
                string out = convert_word(word);

                if (out.length() < 2)
                    continue;

                if (word_dict.find(out) == word_dict.end()) {
                    word_dict[out] = word_index++;
//                    cout << word_index - 1 << " " << out << endl;
                }
            }
//            cout << endl;
            file.close();
        }
    }
    cout << word_dict.size() << endl;
    return word_dict;
}

void build_text_array(fs::path target_dir, map<string,int> dict, int input_dim,
                      int **text_array_out, int **text_class_out, int *text_cnt_out,
                      int *output_dim) {
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
            string word;
//            cout << "on file " << subp << endl;

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
            file.close();
        }
        class_id++;
    }
    *text_array_out = text_array;
    *text_class_out = text_class;
    *text_cnt_out = text_cnt;
    *output_dim = class_id;
}

#define round_to(x, base) ((((x)+(base)-1)*(base)) / (base))
#define BIAS_BASE 2

int attract(int neuron_index, int text_index, double step) {
    assert(text_index < l.text_count);
    assert(neuron_index < l.neuron_count);

    void* args[] = {&l.input_dim, &l.neuron_weight_d, &neuron_index, &l.text_array_d,
                    &text_index, &step};

//    cout << "launching " << round_to(l.input_dim, 1024) / 1024 << " blocks" << endl;
    CHECK_ERROR(cuLaunchKernel(l.attract, round_to(l.input_dim, 1024) / 1024,
            1, 1, 1024, 1, 1, 0, 0, args, 0));
    CHECK_ERROR(cuCtxSynchronize());
}

int compete(int text_index, bool count_bias, double *d)
{
    assert(text_index < l.text_count);

    void* args[] = {&l.input_dim, &l.neuron_weight_d, &l.text_array_d,
                    &text_index,  &l.neuron_dist_d};

    CHECK_ERROR(cuLaunchKernel(l.distance, l.neuron_count,
            1, 1, THREADS_PER_BLOCK, 1, 1, 0, 0, args, 0));
    CHECK_ERROR(cuCtxSynchronize());

    cuMemcpyDtoH(l.neuron_dist, l.neuron_dist_d, sizeof(double) * l.neuron_count);

    double min_dist = numeric_limits<double>::max();
    int min_index = -1;
    for (int i = 0; i < l.neuron_count; i++) {
        double dist = l.neuron_dist[i];
        if (count_bias)
            dist *= pow(BIAS_BASE, l.neuron_bias[i]);
//        cout << "distance of neuron" << i << " is " << dist << endl;
        if (dist <= min_dist) {
            min_index = i;
            min_dist = dist;
        }
    }
    if (min_index == -1) {
        cout << "Warning! Minimum distance neuron not found." << endl;
        return -1;
    }

    if (d != NULL)
        *d = min_dist;
    return min_index;
}

int main() {
    cuInit(0);

    CUdevice device;
    CUcontext context;
    CUmodule cuModule = (CUmodule)0;

    srand(time(NULL));

    CHECK_ERROR(cuDeviceGet(&device, 0));
    CHECK_ERROR(cuCtxCreate(&context, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, device));
    CHECK_ERROR(cuModuleLoad(&cuModule, "lvq.ptx"));

    map<string,int> dict = create_dict(fs::path("./texts"));
    l.input_dim = dict.size();
    l.neuron_count = 4;
    cout << "Created dict. How many neurons to use? ";
    cin >> l.neuron_count;

    cout << "Building text array..." << endl;
    build_text_array(fs::path("./texts"), dict, l.input_dim, &l.text_array,
            &l.text_class, &l.text_count, &l.output_dim);

    cout << l.text_count << " texts in " << l.output_dim << " classes." << endl;
    cout << "Input dimension is " << l.input_dim << endl;

#ifdef DEBUG
    for (int i = 0; i < l.text_count * l.input_dim; i++) {
        cout << l.text_array[i] << " ";
        if (i % l.input_dim == l.input_dim - 1)
            cout << endl;
    }
#endif

    l.neuron_dist = new double[l.neuron_count];
    l.neuron_bias = new int[l.neuron_count];
    l.neuron_class = new int[l.neuron_count];

    for (int i = 0; i < l.neuron_count; i++) {
        l.neuron_bias[i] = 0;
        l.neuron_class[i] = i * l.output_dim / l.neuron_count;
    }

    CHECK_ERROR(cuMemAlloc(&l.neuron_weight_d, l.neuron_count * l.input_dim * sizeof(double)));
    CHECK_ERROR(cuMemAlloc(&l.neuron_dist_d, l.neuron_count * sizeof(double)));
    CHECK_ERROR(cuMemAlloc(&l.text_array_d, l.text_count * l.input_dim * sizeof(int)));

    CHECK_ERROR(cuMemcpyHtoD(l.text_array_d, l.text_array, l.text_count * l.input_dim * sizeof(int)));
    delete[] l.text_array;

    CHECK_ERROR(cuModuleGetFunction(&l.init, cuModule, "init"));
    CHECK_ERROR(cuModuleGetFunction(&l.distance, cuModule, "distance"));
    CHECK_ERROR(cuModuleGetFunction(&l.attract, cuModule, "attract"));

    void* args[] = {&l.input_dim, &l.output_dim, &l.neuron_count,
                    &l.neuron_weight_d};

    CHECK_ERROR(cuLaunchKernel(l.init, l.neuron_count/32,
            1, 1, 32, 1, 1, 0, 0, args, 0));
    CHECK_ERROR(cuCtxSynchronize());

    float attract_step = 0.9, repel_step = -0.5;
    int limit = l.text_count * l.neuron_count;
//    cout << "How many training iterations? ";
//    cin >> limit;
    for (int i = 0; i < limit; i++) {
        int text = rand() % l.text_count;
        attract_step = 0.05 + 0.9 * (1.0 - (double)i/(double)limit);
        repel_step = -0.01 - 0.49 * (1.0 - (double)i/(double)limit);
//        cout << "====================" << endl;
//        cout << "Training text " << text << ", class " << l.text_class[text] << endl;
        double d;
        int winner = compete(text, true, &d);
//        cout << "Winner is " << winner << ", class " << l.neuron_class[winner];
//        cout << ", bias " << l.neuron_bias[winner] <<" distance " << d << endl;
        if (winner == -1)
            break;
        if (l.text_class[text] == l.neuron_class[winner]) {
//            cout << "Attract (step " << attract_step << ")..." << endl;
            attract(winner, text, attract_step);
        } else {
//            cout << "Repel (step " << repel_step << ")..." << endl;
            attract(winner, text, repel_step);
        }
//        winner = compete(text, true, &d);
//        cout << "After training winner is " << winner << " with dist " << d << endl;
        l.neuron_bias[winner] += 1.5;
    }

    int success = 0;
    for (int i = 0; i < l.text_count; i++) {
        int winner = compete(i, false, NULL);
        if (l.text_class[i] == l.neuron_class[winner])
            success++;
    }
    delete[] l.text_class;

    cout << "Random choice accuracy would be " << (1.0 / (float) l.output_dim) << "." << endl;
    cout << "Accuracy on training set " << ((float)success / (float) l.text_count) << "." << endl;

    CHECK_ERROR(cuMemFree(l.text_array_d));

    build_text_array(fs::path("./text_test"), dict, l.input_dim, &l.text_array,
                &l.text_class, &l.text_count, &l.output_dim);

    cout << "New text count is " << l.text_count << endl;
    CHECK_ERROR(cuMemAlloc(&l.text_array_d, l.text_count * l.input_dim * sizeof(int)));
    CHECK_ERROR(cuMemcpyHtoD(l.text_array_d, l.text_array, l.text_count * l.input_dim * sizeof(int)));
    delete[] l.text_array;

    success = 0;
    for (int i = 0; i < l.text_count; i++) {
        int winner = compete(i, false, NULL);
        if (l.text_class[i] == l.neuron_class[winner])
            success++;
    }
    delete[] l.text_class;
    cout << "Accuracy on external set " << ((float)success / (float) l.text_count) << "." << endl;

    cuCtxDestroy(context);

    return 0;
}

