/*
Copyright (c) 2017 Donald Munro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */

#ifndef _TEMPLRANSAC_H_
#define _TEMPLRANSAC_H_

#include <assert.h>

#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>
#include <vector>
#include <tuple>
#include <queue>
#include <unordered_map>
#include <memory>

namespace templransac
{
   inline bool uniform_sample_indices(size_t size, size_t sample_size, std::mt19937& g,
                               std::vector<size_t>& samples, std::vector<size_t>& unsampled)
   //------------------------------------------------------------------------------
   {
      unsampled.resize(size);
      std::iota(unsampled.begin(), unsampled.end(), 0);
      samples.resize(sample_size);
      std::shuffle(unsampled.begin(), unsampled.end(), g);
      std::copy(unsampled.begin(), unsampled.begin() + sample_size, samples.begin());
      unsampled.erase(unsampled.begin(), unsampled.begin() + sample_size);
      return (samples.size() == sample_size);
   }

   struct RANSACParams
   {
      RANSACParams(const double error_thresh_,
                   const double sample_inlier_probability_ = 0.95,
                   const double inlier_probability_ = 0.5) :
                   error_threshold(error_thresh_),
                   sample_inlier_probability(sample_inlier_probability_),
                   inlier_probability(inlier_probability_)
      {
         if ( (sample_inlier_probability <= 0) || (sample_inlier_probability >= 1) )
            std::cerr << "sample_inlier_probability out of range" << std::endl;
         assert( (sample_inlier_probability > 0) && (sample_inlier_probability < 1) );
         if ( (inlier_probability <= 0) || (inlier_probability >= 1) )
            std::cerr << "inlier_probability out of range" << std::endl;
         assert( (inlier_probability > 0) && (inlier_probability < 1) );
      }

      double error_threshold; // Error threshold used to distinguish between inliers and outliers
      double sample_inlier_probability; // The probability that at least one sample is free of outliers
      double inlier_probability; // The probability that a given data item is an inlier

      //=============== Output parameters ===========================
      size_t iterations = 0, inlier_count =0, outlier_count =0;
   };

   template<typename M, typename Q>
   inline void set_best(const double cost, double& best_cost, std::vector<size_t>* best_inliers,
                        const size_t no_results, M& result, std::vector<size_t> inliers, Q& q, M& m)
   //--------------------------------------------------------------------------------------------------
   {
      if ( (no_results == 1) && (cost < best_cost) )
      {
         result = m;
         best_cost = cost;
         *best_inliers = std::move(inliers);
      }
      else if ( (cost < best_cost) || (q.size() < no_results) )
      {
         q.push(std::make_tuple(cost, m, std::move(inliers)));
         if (q.size() > no_results)
            q.pop();
         best_cost = std::get<0>(q.top());
      }
   }

   /**
    * The main RANSAC templated code.
    * @tparam Data A template class acting as a data container container ie it contains containers with the
    * actual data eg for PnP pose the data class could be defined as
    * \code{.cpp}
    * struct PnPData
    *{
    *    std::vector<cv::Point2f> &points2d;
    *    std::vector<cv::Point3f> &points3d;
    *};
    * \endcode
    * @tparam Model The RANSAC model which contains the implementation specific information being optimised.
    * The class should implement a copy constructor and an operator= (unless it is certain the default bitwise copy
    * will do the right thing).
    *
    * @tparam Estimator The RANSAC estimator which contains the implementation specific sample estimation and error
    * calculation. It should implement two methods namely:
    * \code{.cpp}
    * const int estimate(const D &samples, const std::vector<size_t> &sampleIndices,
    *                            RANSACParams &parameters, std::vector<M> &fitted_models) const { return 0; }
    * const void error(const D& samples, const std::vector<size_t>& sampleIndices,
    *                          M& model, std::vector<double>& errors,
    *                          std::vector<size_t>& inlier_indices, std::vector<size_t>& outlier_indices,
    *                          double error_threshold) const {};
    * \endcode
    * The methods are called by template 'duck-typing' so the class does not need to be derived from a base class.
    * In both methods the sampleIndices vector contains indices into the samples data containers which indicate
    * the data items which the methods should use when executing their calculations.
    * The estimate method returns the number of models generated (usually 1 but many pose algorithms solving
    * quadratic or higher polynomials may return more than 1.)
    *
    * @param parameters The RANSAC parameters
    * @param estimator The RANSAC estimator as described in the Estimator template type above.
    * @param data The data container container as described in the Data template type above.
    * @param datasize The size of the data in the Data container
    * @param sample_size The minimum sample size to be used by RANSAC
    * @param no_results The number of results to return (allows multiple solutions to be returned sorted by the cost.
    * @param results The RANSAC results as a vector of pairs of cost and the best Model. . If no_results > 1 then the
    * results are sorted in descending order. In both 1 and > 1 cases the best result in results[0].
    * @param inliers The inliers as a vector of vectors corresponding to the models for the same index in results.
    * @param errs A pointer to a stringstream which, if not null, will contain a human readable form of any errors.
    * @return the success probability of the best result.
    */
   template<typename Data, typename Model, typename Estimator>
   double RANSAC(RANSACParams& parameters, const Estimator estimator, const Data& data, const size_t datasize,
                 const size_t sample_size, const size_t no_results, std::vector<std::pair<double, Model>>& results,
                 std::vector<std::vector<size_t>>& inliers, std::stringstream* errs = nullptr)
   //---------------------------------------------------------------------------------------------------------------
   {
      Model result;
      const auto sized = static_cast<double>(datasize), sample_sized = static_cast<double>(sample_size);
      double best_cost = std::numeric_limits<double>::max();
      std::vector<size_t>* best_inlier_indices;
      if (no_results == 1)
      {
         inliers.emplace_back();
         best_inlier_indices = &inliers[0];
      }
      else
         best_inlier_indices = nullptr;
      const double inlier_sample_probability_log = std::log(1.0 - parameters.sample_inlier_probability);
      double inlier_probability = parameters.inlier_probability;
      auto qcmp = [](std::tuple<double, Model, std::vector<size_t>> left,
                     std::tuple<double, Model, std::vector<size_t>> right)
         { return std::get<0>(left) < std::get<0>(right); };
      std::priority_queue<std::tuple<double , Model, std::vector<size_t>>,
                          std::vector<std::tuple<double, Model, std::vector<size_t>>>, decltype(qcmp)> result_q(qcmp);
      if (datasize < sample_size)
      {
         if (errs != nullptr)
            *errs << "Data datasize " << datasize << " is less than sample datasize " << sample_size;
         return -1;
      }
      const double error_threshold = parameters.error_threshold;
      std::vector<size_t> samples, unsampled, all;
      all.resize(datasize);
      std::iota(all.begin(), all.end(), 0);
      std::vector<Model> estimated_models;
      int cmodels = 0;
      size_t iteration = 0;
      if (datasize == sample_size)
      {
         samples.resize(datasize);
         std::iota(samples.begin(), samples.end(), 0);
         cmodels = estimator.estimate(data, samples, parameters, estimated_models);
         if (cmodels <= 0)
         {
            if (errs != nullptr)
               *errs << "Could not estimate model from single iteration (data datasize " << datasize << " == "
                     << sample_size << ")";
            return -1;
         }
         std::vector<size_t> inlier_indices, outlier_indices;
         for (Model& model : estimated_models)
         {
            estimator.error(data, samples, model, inlier_indices, outlier_indices, error_threshold);
            const double cost = sized - static_cast<double>(inlier_indices.size());
            set_best(cost, best_cost, best_inlier_indices, no_results, result, inlier_indices, result_q, model);
         }
         iteration = 1;
         if (errs != nullptr)
            *errs << "WARNING: Data datasize " << datasize << " == " << sample_size << " is to small to use RANSAC";
      }
      else
      {
         size_t no_iterations = static_cast<size_t>(std::max(inlier_sample_probability_log /
                                                    std::log(1 - std::pow(inlier_probability, sample_sized)),
                                                    1.0));
         std::random_device rd;
         std::mt19937 g(rd());
         for (iteration = 0; iteration < no_iterations; iteration++)
         {
            if (! uniform_sample_indices(datasize, sample_size, g, samples, unsampled))
            {
               parameters.iterations = iteration + 1;
               if (errs != nullptr)
                  *errs << "Error drawing sample of datasize " << sample_size << " from data of datasize " << datasize;
               return -1;
            }
            cmodels = estimator.estimate(data, samples, parameters, estimated_models);
            if (cmodels <= 0)
               continue;
            for (Model& model : estimated_models)
            {
               std::vector<size_t> inlier_indices, outlier_indices;
               estimator.error(data, all, model, inlier_indices, outlier_indices, error_threshold);
               assert((inlier_indices.size() + outlier_indices.size()) == all.size());
               //const double cost = sized - static_cast<double>(inlier_indices.size());
               const double cost = static_cast<double>(outlier_indices.size());
               inlier_probability = std::max(static_cast<double>(inlier_indices.size()) / sized, inlier_probability);
               set_best(cost, best_cost, best_inlier_indices, no_results, result, inlier_indices, result_q, model);
               no_iterations = static_cast<size_t>(inlier_sample_probability_log /
                                                   std::log(1 - std::pow(inlier_probability, sample_sized)));
            }
         }
      }
      if (no_results > 1)
      {
         while (! result_q.empty())
         {
            std::tuple<double , Model, std::vector<size_t>> t = result_q.top();
            results.insert(results.begin(), std::make_pair(std::get<0>(t), std::get<1>(t)));
            inliers.insert(inliers.begin(), std::get<2>(t));
            result_q.pop();
         }
         result = results[0].second;
      }
      else
         results.push_back(std::make_pair(best_cost, result));

      std::vector<size_t> inlier_indices, outlier_indices;
      estimator.error(data, all, result, inlier_indices, outlier_indices, error_threshold);
      inlier_probability = static_cast<double>(inlier_indices.size()) / sized;
      parameters.iterations = iteration;
      parameters.inlier_count = inlier_indices.size();
      parameters.outlier_count = outlier_indices.size();
      return inlier_probability;
//      return (1.0 - (1.0 - std::pow(std::pow(inlier_probability, sample_sized), iteration)));
   }
}


#endif
