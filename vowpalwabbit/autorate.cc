/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
 */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <sstream>
#include <numeric>
#include <vector>

#include "autorate.h"
#include "cache.h"
#include "v_hashmap.h"
#include "vw.h"
#include "rand48.h"

using namespace std;

namespace AUTORATE {

  struct autorate{
    uint32_t B; //number of learning rates to explore in parallel
    uint32_t increment;
    uint32_t total_increment;
    vector<double> pred_vec;
    vector<double> loss_sums;
    double weighted_examples;
    unsigned best_rate;
    learner base;
    vw* all;
  };

  void find_best(autorate* d)
  {
    d->best_rate = 1;
    double best_loss = d->loss_sums[1];
    for(int i=2; i<= 1+2*d->B;i++)
    {
      if(best_loss > d->loss_sums[i]) 
      {
        best_loss = d->loss_sums[i];
        d->best_rate = i;
      }  
    }
  }

  float weight_gen(size_t i)//sampling from Poisson with rate 1
  { 
    // 2^0 1/2^1 2^1 1/2^2 2^2 1/2^3 2^3 

    int exp = i / 2;
    size_t mod = i % 2;

    double temp = pow(2.0,exp);

    if(mod == 0) temp=1.0/temp;
    cout << "#weight_gen\ti=" << i << "\texp=" << exp << "\tmod=" << mod << "\ttemp=" << temp << "\n";
    if(i == 1) return 1;
    return temp;
  }

  void print_result(int f, float res, float weight, v_array<char> tag)
  {
    if (f >= 0)
    {
      char temp[30];
      sprintf(temp, "%f", res);
      std::stringstream ss;
      ss << temp;
      print_tag(ss, tag);
      ss << '\n';
      ssize_t len = ss.str().size();
#ifdef _WIN32
	  ssize_t t = _write(f, ss.str().c_str(), (unsigned int)len);
#else
	  ssize_t t = write(f, ss.str().c_str(), (unsigned int)len);
#endif
      if (t != len)
        cerr << "write error" << endl;
    }    
  }

  void output_example(vw& all, example* ec, autorate* d)
  {
    if (command_example(&all,ec))
      return;

    label_data* ld = (label_data*)ec->ld;


   if(ec->test_only)
    {
      all.sd->weighted_holdout_examples += ld->weight;//test weight seen
      all.sd->weighted_holdout_examples_since_last_dump += ld->weight;
      all.sd->weighted_holdout_examples_since_last_pass += ld->weight;
      all.sd->holdout_sum_loss += ec->loss;
      all.sd->holdout_sum_loss_since_last_dump += ec->loss;
      all.sd->holdout_sum_loss_since_last_pass += ec->loss;//since last pass
    }
    else
    {
      all.sd->weighted_examples += ld->weight;
      all.sd->sum_loss += ec->loss;
      all.sd->sum_loss_since_last_dump += ec->loss;
      all.sd->total_features += ec->num_features;
      all.sd->example_number++;
    }

    for (int* sink = all.final_prediction_sink.begin; sink != all.final_prediction_sink.end; sink++)
      AUTORATE::print_result(*sink, ec->final_prediction, 0, ec->tag);
  
    print_update(all, ec);
  }

  void print_autorates(autorate* d)
  {
    cout << "autorates:\n";
    for(unsigned i=1; i<=1+2*d->B ; i++)
    {
      cout << "i=" << i << "\tiw=" << weight_gen(i) << "\tloss_sum=" << d->loss_sums[i] << "\tloss_mean=" << (d->loss_sums[i]/d->weighted_examples) << "\n";
    }
  }

  void learn_with_output(autorate* d, example* ec, bool shouldOutput)
  {
    vw* all = d->all;
    if (command_example(all,ec))
      {
	d->base.learn(ec);
	return;
      }

    double weight_temp = ((label_data*)ec->ld)->weight;
  
    string outputString;
    stringstream outputStringStream(outputString);
    d->pred_vec.clear();

    for (size_t i = 1; i <= 1+2*d->B; i++)
      {
        if (i != 1)
          update_example_indicies(all->audit, ec, d->increment);
        
        double weight_mod = weight_gen(i);
          
        ((label_data*)ec->ld)->weight = weight_temp * weight_mod;

        d->base.learn(ec);

        d->pred_vec.push_back(ec->final_prediction);
        d->loss_sums[i] += all->loss->getLoss(all->sd, ec->final_prediction, ((label_data*)ec->ld)->label) * weight_mod;

        if (shouldOutput) {
          if (i > 1) outputStringStream << ' ';
          outputStringStream << i << ':' << ec->partial_prediction;
        }
      }	
    d->weighted_examples += weight_temp;

    print_autorates(d);  

    ((label_data*)ec->ld)->weight = weight_temp;

    update_example_indicies(all->audit, ec, -d->total_increment);

    ec->final_prediction = d->pred_vec[0];
    ec->loss = all->loss->getLoss(all->sd, ec->final_prediction, ((label_data*)ec->ld)->label) * ((label_data*)ec->ld)->weight;

    if (shouldOutput) 
      all->print_text(all->raw_prediction, outputStringStream.str(), ec->tag);

  }

  void learn(void* d, example* ec) {
    learn_with_output((autorate*)d, ec, false);
  }

  void drive(vw* all, void* d)
  {
    example* ec = NULL;
    while ( true )
      {
        if ((ec = VW::get_example(all->p)) != NULL)//semiblocking operation.
          {
            learn_with_output((autorate*)d, ec, all->raw_prediction > 0);
            if (!command_example(all, ec))
              AUTORATE::output_example(*all, ec, (autorate*)d);
	    VW::finish_example(*all, ec);
          }
        else if (parser_done(all->p))
	  return;
        else 
          ;
      }
  }

  void finish(void* data)
  {    
    autorate* o=(autorate*)data;
    o->base.finish();
    free(o);
  }

  learner setup(vw& all, std::vector<std::string>&opts, po::variables_map& vm, po::variables_map& vm_file)
  {
    autorate* data = (autorate*)calloc(1, sizeof(autorate));

    po::options_description desc("BS options");
    desc.add_options()
      ("autorate_base", po::value<string>(), "will explore rates that are powers of this number and reciprocals");

    po::parsed_options parsed = po::command_line_parser(opts).
      style(po::command_line_style::default_style ^ po::command_line_style::allow_guessing).
      options(desc).allow_unregistered().run();
    opts = po::collect_unrecognized(parsed.options, po::include_positional);
    po::store(parsed, vm);
    po::notify(vm);

    po::parsed_options parsed_file = po::command_line_parser(all.options_from_file_argc,all.options_from_file_argv).
      style(po::command_line_style::default_style ^ po::command_line_style::allow_guessing).
      options(desc).allow_unregistered().run();
    po::store(parsed_file, vm_file);
    po::notify(vm_file);

    if( vm_file.count("autorate") ) {
      data->B = (uint32_t)vm_file["autorate"].as<size_t>();
      if( vm.count("autorate") && (uint32_t)vm["autorate"].as<size_t>() != data->B )
        std::cerr << "warning: you specified a different number of samples through --autorate than the one loaded from predictor. Pursuing with loaded value of: " << data->B << endl;
    }
    else {
      data->B = (uint32_t)vm["autorate"].as<size_t>();

      //append autorate with number of samples to options_from_file so it is saved to regressor later
      std::stringstream ss;
      ss << " --autorate " << data->B;
      all.options_from_file.append(ss.str());
    }

    data->best_rate = 1;
    data->pred_vec.reserve(1+2*data->B);
    data->loss_sums.reserve(1+2*data->B);
    data->all = &all;
    data->increment = all.reg.stride * all.weights_per_problem;
    all.weights_per_problem *= 1+2*data->B;
    data->total_increment = data->increment*(1+2*data->B-1);
    data->base = all.l;
    learner l(data, drive, learn, finish, all.l.sl);
    return l;
  }
}
