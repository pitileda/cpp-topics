#pragma once

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <vector>

namespace sdl {

class Zoo {
private:
	std::promise<void> run_done_, walk_done_, tell_done_;
	std::atomic_bool m_shutdown;
	std::vector<int> v_;
public:
	Zoo(){
		m_shutdown.store(false);
		v_.reserve(10);
		v_.push_back(12);
	}
	~Zoo(){
		std::cout <<"Waiting for all methods to be done\n";
		m_shutdown.store(true);
		// run_done_.get_future().get();
		// walk_done_.get_future().get();
		// tell_done_.get_future().get();
		std::cout <<"all methods are done\n";
	}

	void run(){
		if(m_shutdown.load()){
			run_done_.set_value();
			return;
		}
		run_done_ = std::promise<void>();
		while(!m_shutdown.load()){
			v_.push_back(0);
		}
		run_done_.set_value();
	}

	void walk(){
		if(m_shutdown.load()){
			walk_done_.set_value();
			return;
		}
		walk_done_ = std::promise<void>();
		while(!m_shutdown.load()){
			v_.push_back(0);
		}
		walk_done_.set_value();
	}

	void tell(){
		if(m_shutdown.load()){
			tell_done_.set_value();
			return;
		}
		tell_done_ = std::promise<void>();
		while(!m_shutdown.load()){
			v_.push_back(0);
		}
		tell_done_.set_value();
	}

	void threadFuncOne(){
		while(!m_shutdown.load()){
			run();
			walk();
			tell();
		}
	}

	void threadFuncTwo(){
		std::chrono::milliseconds time_interval(1000);
		std::this_thread::sleep_for(time_interval);
		set();
	}

	void set() {
		m_shutdown.store(true);
	}

};

} //namespace prms