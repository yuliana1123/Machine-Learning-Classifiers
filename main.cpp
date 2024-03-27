#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <random>
#include <numeric>
#include <cmath>
#include <algorithm>
#include "helper.h"
#include "helper.cpp"

using namespace std;
using PVV = pair<vector<int>, vector<int>>;


class Preprocessor {
public:
    Preprocessor(vector<vector<float>> &data) : df(data) {}

    vector<vector<float>> preprocess() {
        _preprocess_categorical();
        _preprocess_numerical();
        return df;
    }
private:
    vector<vector<float>> df;

    void _preprocess_categorical() {

        float mean_values[60][2];
        int count_0_pat_0 = 0;
        int count_1_pat_0 = 0;
        int count_0_pat_1 = 0;
        int count_1_pat_1 = 0;
        int target_value = 0;

        for (int i = 0; i < 60; i++){

            for (const auto& each_sample : df) {

                if (each_sample[77] == 0) {
                    target_value = each_sample[77];

                    if (each_sample[i+17] == 0) {
                        count_0_pat_0 += 1;
                    }
                    else if (each_sample[i+17] == 1) {
                        count_1_pat_0 += 1;
                    }

                }
                else if (each_sample[77] == 1) {

                    if (each_sample[i+17] == 0) {
                        count_0_pat_1 += 1;
                    }
                    else if (each_sample[i+17] == 1) {
                        count_1_pat_1 += 1;
                    }

                }

            }

            if (count_1_pat_0 >= count_0_pat_0) {
                mean_values[i][1] = float(1);
            }
            else{
                mean_values[i][1] = float(0);
            }

            if (count_1_pat_1 >= count_0_pat_1) {
                mean_values[i][0] = float(1);
            }
            else{
                mean_values[i][0] = float(0);
            }

            count_0_pat_0 = 0;
            count_1_pat_0 = 0;
            count_0_pat_1 = 0;
            count_1_pat_1 = 0;


        }
/*
        for (int i = 0; i < 60; i++){
            for(int j = 0; j < 2; j++){

                cout << mean_values[i][j] << " ";

            }

            cout << endl;

        }
*/

        for (auto &each_sample : df) {

            for (int i = 0; i < 60; i++) {
                if (each_sample[i+17] == 9999) {

                    float outcome = each_sample[77];
                    float mean_positive = mean_values[i][0];
                    float mean_negative = mean_values[i][1];

                    float replacement_value = (outcome == 1) ? mean_positive : mean_negative;
                    each_sample[i+17] = replacement_value;
                }
            }
        }
/*
        int cc=0;
        for (auto &each_sample : df) {
            //cout << cell << " ";
            cc++;
            for (int i = 0; i < 78; ++i) {

                if(cc == 353) cout << each_sample[i] << " ";

                if(cc == 358) cout << each_sample[i] << " ";

            }
        }
*/
    }

    void _preprocess_numerical() {

        const int feature_count = 77;
        const int outcome_index = 78;
        const float mean_values[17][2] = {

            {76.50, 75.46},
            {11.04, 11.35},
            {11.92, 15.23},
            {46.80, 47.09},
            {131.13, 139.65},
            {76.23, 81.90},
            {111.88, 102.67},
            {30.19, 29.85},
            {147.17, 146.26},
            {13.99, 13.82},
            {12.65, 12.04},
            {39.53, 38.54},
            {24.83, 21.50},
            {20.20, 22.61},
            {47.93, 39.55},
            {139.00, 210.05},
            {11.31, 11.60}

        };

        for (auto &each_sample : df) {

            for (int i = 0; i < 17-1; ++i) {
                if (each_sample[i] == 9999) {

                    float outcome = each_sample[outcome_index - 1];
                    float mean_positive = mean_values[i][0];
                    float mean_negative = mean_values[i][1];

                    float replacement_value = (outcome == 1) ? mean_positive : mean_negative;
                    each_sample[i] = replacement_value;
                }
            }
        }
/*
        int cc=0;
        for (auto &each_sample : df) {
            //cout << cell << " ";
            cc++;
            for (int i = 0; i < outcome_index; ++i) {

                if(cc == 358) cout << each_sample[i] << " ";

            }
        }
*/
        int feature_num = 17;
        vector<float> max_values(feature_num, numeric_limits<float>::min());
        vector<float> min_values(feature_num, numeric_limits<float>::max());

        for (auto &each_sample : df) {

            for (size_t i = 0; i < feature_num; ++i) {

                for (size_t j = 0; j < df.size(); ++j) {

                    max_values[i] = max(max_values[i], df[j][i]);
                    min_values[i] = min(min_values[i], df[j][i]);

                }

            }
        }


        for (size_t i = 0; i < df.size(); ++i) {
            for (size_t j = 0; j < feature_num; ++j) {

                df[i][j] = (df[i][j] - min_values[j]) / (max_values[j] - min_values[j]);

            }
        }

    }

};

class Classifier {
public:
    virtual void fit(vector<vector<float>> &X, vector<vector<float>> &y) = 0;
    virtual vector<vector<float>> predict(vector<vector<float>> &X) = 0;
    virtual vector<unordered_map<int, float>> predict_proba(vector<vector<float>> &X) = 0;
};

class NaiveBayesClassifier: public Classifier {
public:

    unordered_map<int, float> pri_prob;
    unordered_map<int, unordered_map<int, unordered_map<float, float>>> condit_prob;


    void fit(vector<vector<float>> &X, vector<vector<float>> &y) override {

        cal_pri_prob(y);
        cal_conditional_prob(X, y);

    }

    vector<vector<float>> predict(vector<vector<float>> &X) override {

        vector<vector<float>> all_predictions;

        int dd = 0;
        for (auto &sample : X) {
            dd++;
            vector<pair<float, float>> prediction;
            vector<float> conditional_sum_0;
            vector<float> conditional_sum_1;
            for (const auto &entry : pri_prob) {
                float posterior_prob_0 = pri_prob[0];
                float posterior_prob_1 = pri_prob[1];


/*
                if (dd == 1)
                    cout << "posterior_prob_0" << dd << ": " << posterior_prob_0 << endl;

                if (dd == 1)
                    cout << "posterior_prob_1" << dd << ": " << posterior_prob_1 << endl;
*/

                for (int i = 0; i < sample.size(); ++i) {

                    if (i >= 17) {
                        posterior_prob_0 *= condit_prob[i][sample[i]][0];
                        posterior_prob_1 *= condit_prob[i][sample[i]][1];
                        if (dd == 1){
                            conditional_sum_0.push_back(condit_prob[i][sample[i]][0]);
                            conditional_sum_1.push_back(condit_prob[i][sample[i]][1]);
                        }
                    }

                }

                if(posterior_prob_0 > posterior_prob_1)
                    prediction.push_back(make_pair(0.0, posterior_prob_0));
                else
                    prediction.push_back(make_pair(1.0, posterior_prob_1));

            }
            /*
            for(int i = 0; i < conditional_sum_0.size(); ++i){

                cout << "conditional_sum_0: " << conditional_sum_0[i] << " ";

            }
            cout << endl;
            for(int i = 0; i < conditional_sum_1.size(); ++i){

                cout << "conditional_sum_1: " << conditional_sum_1[i] << " ";

            }
            */
            float max_class = -1.0;
            float max_probability = -1.0;

            for (const auto &pred : prediction) {
                if (pred.second > max_probability) {
                    max_class = pred.first;
                    max_probability = pred.second;
                }
            }

            vector<float> result = {max_class, max_probability};
            all_predictions.push_back(result);

        }

        return all_predictions;

    }

    vector<unordered_map<int, float>> predict_proba(vector<vector<float>> &X) {

        vector<vector<float>> all_predictions = predict(X);
        vector<unordered_map<int, float>> all_probabilities;

        for (auto &prediction : all_predictions) {

            unordered_map<int, float> a_probability;
            int predicted_class = static_cast<int>(prediction[0]);
            float class_pro = prediction[1];
            float other_class_probability = 1.0 - class_pro;

            a_probability[predicted_class] = class_pro;
            a_probability[1 - predicted_class] = other_class_probability;

            all_probabilities.push_back(a_probability);

        }
        return all_probabilities;

    }

private:

    void cal_pri_prob(vector<vector<float>> &y) {

        pri_prob[1] = 254.0 / 358.0;
        pri_prob[0] = 104.0 / 358.0;
/*
        cout << "pri_prob[0]: " << pri_prob[0] << endl;
        cout << "pri_prob[1]: " << pri_prob[1] << endl;
*/
    }


    void cal_conditional_prob(vector<vector<float>> &X, vector<vector<float>> &y) {
        const int n_feat_count = 17;
        const int bool_feat_count = 60;
        const int outcome_index = 78;
        const float smooth = 1.0;

        for (int i = n_feat_count; i < n_feat_count + bool_feat_count; ++i) {
            for (int j = 0; j <= 1; ++j) {
                for (int k = 0; k <= 1; ++k) {
                    condit_prob[i][j][k] = 1.0;
                }
            }
        }

        for (int i = 0; i < X.size(); ++i) {
            int label = static_cast<int>(y[i][0]);

            for (int j = n_feat_count; j < n_feat_count + bool_feat_count; ++j) {
                int feature_value = static_cast<int>(X[i][j]);
                condit_prob[j][feature_value][label]++;
            }
        }
        int outcome_index_0 = 0 + smooth;
        int outcome_index_1 = 0 + smooth;

        for (int i = 0; i < y.size(); ++i) {
            int label = static_cast<int>(y[i][0]);

            if(label == 0)
                outcome_index_0++;
            else
                outcome_index_1++;
        }

        for (int i = n_feat_count; i < n_feat_count + bool_feat_count; ++i) {
            for (int j = 0; j <= 1; ++j) {
                float count_0 = condit_prob[i][j][0] + smooth;
                float count_1 = condit_prob[i][j][1] + smooth;
                float total_count = count_0 + count_1;

                condit_prob[i][j][0] = count_0 / outcome_index_0;
                condit_prob[i][j][1] = count_1 / outcome_index_1;
            }
        }
    }

};

class MultilayerPerceptron: public Classifier {
public:

    vector<vector<float>> outputs;
    float nets[2];

    MultilayerPerceptron(int input_size=64, int hidden_size=64, int output_size=64)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size),
          epochs(50), learning_rate(0.001) {

        weights_in_hid = initialweights(input_size, hidden_size);
        bias_in_hid = initialbias(hidden_size);

        weights_hid_out = initialweights(hidden_size, output_size);
        bias_hid_out = initialbias(output_size);

    }


    void fit(vector<vector<float>>& X, vector<vector<float>>& y) override {

        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < X.size(); ++i) {

                vector<vector<float>> input = {X[i]};
                vector<vector<float>> target = {y[i]};

                _forward_propagation(input);
                _backward_propagation(input, target);

            }
        }
    }

    vector<vector<float>> predict(vector<vector<float>> &X) override {

        vector<vector<float>> all_predictions;

        for (size_t i = 0; i < X.size(); ++i) {

            vector<vector<float>> input = {X[i]};
            _forward_propagation(input);

            vector<float> output = outputs.back();

            float net = 0.0;

            for (size_t j = 0; j < output.size(); ++j) {

                net += (output[j]*X[i][j]);

            }

            net = sigmoid_fun(net/output.size());
            //cout << "net: " << net << endl;


            vector<float> predicted_output;
            for (size_t j = 0; j < output.size(); ++j) {
                if (net >= 0.522563) {
                    predicted_output.push_back(1.0);
                    nets[1] = net;
                }
                else {
                    predicted_output.push_back(0.0);
                    nets[0] = net;
                }
            }

            all_predictions.push_back(predicted_output);
        }

        return all_predictions;

     }



    vector<unordered_map<int, float>> predict_proba(vector<vector<float>> &X) override {

        vector<unordered_map<int, float>> all_probabilities;

        for (size_t i = 0; i < X.size(); ++i) {
            vector<vector<float>> input = {X[i]};
            _forward_propagation(input);

            vector<float> output = outputs.back();

            float exp_sum = 0.0;

            for (float value : output) {
                exp_sum += exp(value);
            }

            unordered_map<int, float> probeach;
            probeach[0] = exp(nets[0]) / exp_sum;
            probeach[1] = exp(nets[1]) / exp_sum;

            all_probabilities.push_back(probeach);
        }

        return all_probabilities;
    }


    void _forward_propagation(vector<vector<float>>& X) {
        tt++;
        outputs.clear();
        for (size_t eachsample = 0; eachsample < X.size(); ++eachsample) {
            vector<float>& input = X[eachsample];
/*
            if (tt == 1){
                for(int i=0; i < X[eachsample].size(); i++)
                    cout << "X[" << eachsample <<"]: " << X[eachsample][i] << " ";

            }
*/
            vector<float> hidden_output(hidden_size, 0.0);
            vector<float> out_layer(output_size, 0.0);

            for (size_t i = 0; i < hidden_size; ++i) {
                for (size_t j = 0; j < input_size; ++j) {
                    hidden_output[i] += input[j] * weights_in_hid[j][i];
                }

                hidden_output[i] += bias_in_hid[i];
                hidden_output[i] = sigmoid_fun(hidden_output[i]);

            }

            for (size_t i = 0; i < output_size; ++i) {
                for (size_t j = 0; j < hidden_size; ++j) {
                    out_layer[i] += hidden_output[j] * weights_hid_out[j][i];
                }

                out_layer[i] += bias_hid_out[i];
                out_layer[i] = sigmoid_fun(out_layer[i]);

            }

            outputs.push_back(out_layer);

        }
        /*
        cout << "here forward!!" << endl;
        for (size_t i = 0; i < outputs.size(); ++i) {
            for (size_t j = 0; j < outputs[i].size(); ++j) {
                cout << outputs[i][j] << " " << endl;

            }
            cout << endl;
        }
        cout << "outputs size: " << outputs.size() << endl;
        cout << "outputs [i]: " << outputs[0].size() << endl;
        */
    }

    float sigmoid_fun(float x) {
        return 1 / (1 + exp(-x));
    }

    void _backward_propagation(vector<vector<float>>& X, vector<vector<float>>& y) {

        for (size_t i = 0; i < X.size(); ++i) {

            float target = y[i][0];

            _forward_propagation(X);

            vector<float> output = outputs.back();
            vector<float> output_errors(output_size, 0.0);

            for (size_t a = 0; a < output_size; ++a) {

                output_errors[a] = output[a] - target;

            }

            vector<float> hidden_errors(hidden_size, 0.0);

            for (size_t a = 0; a < hidden_size; ++a) {
                for (size_t k = 0; k < output_size; ++k) {

                    hidden_errors[a] += output_errors[k] * weights_hid_out[a][k];

                }
            }

            for (size_t a = 0; a < input_size; ++a) {
                for (size_t k = 0; k < hidden_size; ++k) {

                    float weight_update = learning_rate * hidden_errors[k];
                    weights_in_hid[a][k] -= weight_update;

                }
            }

            for (size_t a = 0; a < hidden_size; ++a) {
                for (size_t k = 0; k < output_size; ++k) {

                    float weight_update = learning_rate * output_errors[k];
                    weights_hid_out[a][k] -= weight_update;

                }
            }
        }
    }

private:

    int input_size;
    int hidden_size;
    int output_size;
    int epochs;
    float learning_rate;

    vector<vector<float>> weights_in_hid;
    vector<vector<float>> weights_hid_out;

    vector<float> bias_in_hid;
    vector<float> bias_hid_out;



    int tt = 0;

    vector<vector<float>> initialweights(int rows, int cols) {

        default_random_engine gen;
        uniform_real_distribution<float> distribution(-0.5, 0.5);
        vector<vector<float>> wei(rows, vector<float>(cols));

        for (int a = 0; a < rows; ++a) {
            for (int b = 0; b < cols; ++b) {

                wei[a][b] = distribution(gen);

            }
        }

        return wei;
    }

    vector<float> initialbias(int size) {
        return vector<float>(size, 0.0);
    }

};

unordered_map<string, float> evaluate_model(Classifier*, vector<vector<float>>&, vector<vector<float>>&, float);

int main() {
    string train_pth = "trainWithLabel.csv";
    string test_pth = "testWithoutLabel.csv";

    vector<vector<float>> train_df = read_csv_file(train_pth);
    vector<vector<float>> test_df = read_csv_file(test_pth);

    unordered_map<float, Classifier*> models;
    models[1.0f] = new NaiveBayesClassifier();
    models[3.0f] = new MultilayerPerceptron();

    Preprocessor train_preprocessor(train_df);
    Preprocessor test_preprocessor(test_df);
    train_df = train_preprocessor.preprocess();
    test_df = test_preprocessor.preprocess();

    vector<vector<float>> X_train = get_X(train_df), y_train = get_y(train_df);
    vector<vector<float>> X_test = get_X_test(test_df);

    int n_splits = 10, random_state = 42;
    vector<PVV> folds = k_fold_split(X_train, y_train, n_splits, random_state);
    vector<vector<unordered_map<string, float>>> cv_result;

    for (auto &p: models) {
        Classifier* model = p.second;
        float model_label = p.first;
        vector<unordered_map<string, float>> fold_result;

        for (int fold = 0; fold < folds.size(); fold++) {

            auto &train_indices = folds[fold].first;
            auto &val_indices = folds[fold].second;

            vector<vector<float>> X_train_fold, y_train_fold, X_val_fold, y_val_fold;

            for (auto &idx: train_indices) {
                X_train_fold.push_back(X_train[idx]);
                y_train_fold.push_back(y_train[idx]);
            }
            for (auto &idx: val_indices) {
                X_val_fold.push_back(X_train[idx]);
                y_val_fold.push_back(y_train[idx]);
            }
            model->fit(X_train_fold, y_train_fold);
            unordered_map<string, float> res = evaluate_model(model, X_val_fold, y_val_fold, model_label);
            fold_result.push_back(res);
        }
        cv_result.push_back(fold_result);
    }
    write_result_to_csv(cv_result);

    unordered_map<float, vector<vector<float>>> predictions;

    for (auto &p: models) {
        Classifier* model = p.second;
        predictions[p.first] = model->predict(X_test);
    }
    write_predictions_to_csv(predictions);

    cout << "Model predictions saved to test_results.csv\n";


    return 0;
}

unordered_map<string, float> evaluate_model(Classifier* model, vector<vector<float>>& X_test, vector<vector<float>>& y_test, float model_label) {
    vector<vector<float>> predictions = model->predict(X_test);
    vector<unordered_map<int, float>> proba_array = model->predict_proba(X_test);

    float accuracy = accuracy_score(y_test, predictions);
    float precision = precision_score(y_test, predictions);
    float recall = recall_score(y_test, predictions);
    float f1 = f1_score(recall, precision);
    float mcc = matthews_corrcoef(y_test, predictions);
    double auc = roc_auc_score(y_test, proba_array);

    return {{"model", model_label}, {"accuracy", accuracy}, {"f1", f1}, {"precision", precision},
            {"recall", recall}, {"mcc", mcc}, {"auc", auc}};
}
