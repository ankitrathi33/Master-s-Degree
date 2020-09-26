
#include "pch.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>


// Remy

// EXERCISE 1 ------------------------------------------------------------
void f(std::string param) {
	for (;;) {
		std::cout << "Function - Thread ID: " << std::this_thread::get_id() << std::endl;
		std::cout << "Function - Param: " << param << std::endl;
	}
}

class F{
public:
	static void f(std::string param) {
		for (;;) {
			std::cout << "Class method - Thread ID: " << std::this_thread::get_id() << std::endl;
			std::cout << "Class method - Param: " << param << std::endl;
		}
	}
};

struct fo {
	fo(){}
	void operator()(std::string param) {
		for (;;) {
			std::cout << "Function object - Thread ID: " << std::this_thread::get_id() << std::endl;
			std::cout << "Function object - Param: " << param << std::endl;
		}
	}
};


void exercise1(std::string param) {
	fo functor = fo();

	std::thread t1(f, param);
	std::thread t2([](std::string param) -> void {
		for (;;) {
			std::cout << "Lambda - Thread ID: " << std::this_thread::get_id() << std::endl;
			std::cout << "Lambda - Param: " << param << std::endl;
		}
	}, param);
	std::thread t3(functor, param);
	std::thread t4(F::f, param);

	t1.join();
	t2.join();
	t3.join();
	t4.join();
}

// EXERCISE 2 ------------------------------------------------------------
std::mutex m;

void print(std::string text) {
	m.lock();
	std::cout << text << std::endl;
	m.unlock();
}

void exercise2(std::string param) {
	std::vector<std::thread> threads;

	for (int i = 0; i < 20; i++)
		threads.emplace_back(print, param);

	for (auto& t : threads)
		t.join();
}

// EXERCISE 3 -------------------------------------------------------------
typedef std::chrono::high_resolution_clock::time_point time_var;

#define duration(a) std::chrono::duration_cast<std::chrono::milliseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()



std::atomic_int I = 0;

void increment() {
	for (int i = 0; i < 10000000; i++)
		I++;
}

void exercise3(std::string param) {
	std::vector<std::thread> threads;

	time_var start = timeNow();
	increment();
	std::cout << "Single thread time : " << duration(timeNow() - start) << std::endl;

	start = timeNow();
	for (int i = 0; i < 10; i++)
		threads.emplace_back(increment);

	for (auto& t : threads)
		t.join();
	std::cout << "10 threads time : " << duration(timeNow() - start) << std::endl;
}



// MAIN -------------------------------------------------------------
int main(){
	//exercise1("text");
	//exercise2("text");
	//exercise3("text");
}