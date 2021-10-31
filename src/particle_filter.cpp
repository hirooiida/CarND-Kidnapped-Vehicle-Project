/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
    
  if (initialized()) {
    return;
  }
  
  num_particles = 100;  // TODO: Set the number of particles
  
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  std::default_random_engine engine;
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(engine);
    p.y = dist_y(engine);
    p.theta = dist_theta(engine);
    p.weight = 1.0;
    
    particles.push_back(p);
    weights.push_back(p.weight);
  }
  
  is_initialized = true;
  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  std::normal_distribution<double> dist_x(0.0, std_pos[0]);
  std::normal_distribution<double> dist_y(0.0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0.0, std_pos[2]);
  
  std::default_random_engine engine;
  for (auto &p : particles) {
        
    if (fabs(yaw_rate) < 0.0001) {
      p.x += velocity * delta_t * std::cos(p.theta);
      p.y += velocity * delta_t * std::sin(p.theta);
    } else {
      p.x += (velocity / yaw_rate) * (std::sin(p.theta + yaw_rate * delta_t) - std::sin(p.theta));
      p.y += (velocity / yaw_rate) * (std::cos(p.theta) - std::cos(p.theta + yaw_rate * delta_t));
    }
    
    p.theta += yaw_rate * delta_t;
    
    p.x += dist_x(engine);
    p.y += dist_y(engine);
    p.theta += dist_theta(engine);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  for (auto &o : observations) {
    
    double tmp_min_distance = 1000000.0;
    int tmp_id = -1;
    
    for (auto &p : predicted) {
      double cur_distance = dist(o.x, o.y, p.x, p.y);
      
      if (cur_distance < tmp_min_distance) {
        tmp_min_distance = cur_distance;
        tmp_id = p.id;
      }
    }
    
    o.id = tmp_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  double weights_sum = 0.0;
  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
  
  for (auto &p : particles) {
    
    vector<LandmarkObs> predicted;
    for (unsigned int i = 0; i < map_landmarks.landmark_list.size(); i++) {
      LandmarkObs obs;
      obs.id = map_landmarks.landmark_list[i].id_i;
      obs.x = map_landmarks.landmark_list[i].x_f;
      obs.y = map_landmarks.landmark_list[i].y_f;
      
      if (dist(obs.x, obs.y, p.x, p.y) < sensor_range) {
        predicted.push_back(obs);
      }
    }
    
    vector<LandmarkObs> trans_observations;
    for (unsigned int i = 0; i < observations.size(); i++) {
      LandmarkObs trans_obs;
      LandmarkObs cur_obs = observations[i];
      
      trans_obs.id = cur_obs.id;
      trans_obs.x = p.x + cur_obs.x * std::cos(p.theta) - cur_obs.y * std::sin(p.theta);
      trans_obs.y = p.y + cur_obs.x * std::sin(p.theta) + cur_obs.y * std::cos(p.theta);
      
      trans_observations.push_back(trans_obs);
    }
    
    dataAssociation(predicted, trans_observations);
    
    for (unsigned int i = 0; i < trans_observations.size(); i++) {
      
      double trans_x = trans_observations[i].x;
      double trans_y = trans_observations[i].y;
      int trans_id = trans_observations[i].id;
      
      double pred_x;
      double pred_y;
      p.weight = 1.0;
      
      for (unsigned int j = 0; j < predicted.size(); j++) {
        
        int pred_id = predicted[j].id;

        if (pred_id == trans_id) {
          pred_x = predicted[j].x;
          pred_y = predicted[j].y;
        }
      }
      double exponent = std::pow(pred_x - trans_x, 2) / (2 * std::pow(sig_x,2)) + std::pow(pred_y - trans_y, 2) / (2 * std::pow(sig_y, 2));
      p.weight *= gauss_norm * std::exp(-exponent);
      weights[p.id] = p.weight;
      weights_sum += p.weight;
    }
  }
  
  for (int i = 0; i < num_particles; i++) {
    particles[i].weight /= weights_sum;
    weights[i] = particles[i].weight;
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  vector<Particle> new_particles;
  
  double max_weight = 0.0;
  for (const auto &w : weights) {
    if (w > max_weight) {
      max_weight = w;
    }
  }
  
  std::default_random_engine engine;
  std::uniform_real_distribution<double> dist_d(0.0, max_weight);
  std::uniform_int_distribution<int> dist_i(0, num_particles - 1);
  
  int index = dist_i(engine);
  double beta = 0.0;
  
  for (int i = 0; i < num_particles; i++) {
    beta += dist_d(engine) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
