#include "pch.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>


// Suar Remy

// EXERCISE 1 ------------------------------------------------------------
void f(std::string param) {
	for (;;) {
		std::cout << "Function - Thread ID: " << std::this_thread::get_id() << std::endl;
		std::cout << "Function - Param: " << param << std::endl;
	}
}

class F {
public:
	static void f(std::string param) {
		for (;;) {
			std::cout << "Class method - Thread ID: " << std::this_thread::get_id() << std::endl;
			std::cout << "Class method - Param: " << param << std::endl;
		}
	}
};

struct fo {
	fo() {}
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

#define duration(a) std::chrono::duration_cast<std::chrono::microseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()



std::atomic_int atomic_i = 0;
int I1 = 0;
int I2 = 0;
std::mutex m2;

void increment() {
	for (size_t i = 0; i < 100000; i++)
	{
		I1++;
	}
}

void atomicIncrement() {
	for (size_t i = 0; i < 100000; i++)
	{
		atomic_i++;
	}
}

void mutexIncrement() {
	for (size_t i = 0; i < 100000; i++)
	{
		m2.lock();
		I2++;
		m2.unlock();
	}
}

void exercise3() {
	std::vector<std::thread> threads;
	std::vector<std::thread> threads3;
	std::vector<std::thread> threads2;

	// One thread
	time_var start = timeNow();
	for (int i = 0; i < 10; i++)
		threads3.emplace_back(increment);

	for (auto& t : threads3)
		t.join();
	std::cout << "10 threads time : " << duration(timeNow() - start) << " us" << std::endl;

	// 10 threads - atomic
	start = timeNow();
	for (int i = 0; i < 10; i++)
		threads.emplace_back(atomicIncrement);

	for (auto& t : threads)
		t.join();
	std::cout << "10 threads time atomic increment : " << duration(timeNow() - start) << " us" << std::endl;

	// 10 threads - mutex
	start = timeNow();
	for (int i = 0; i < 10; i++)
		threads2.emplace_back(mutexIncrement);

	for (auto& t : threads2)
		t.join();
	std::cout << "10 threads time mutex increment: " << duration(timeNow() - start) << " us" << std::endl;
}



// MAIN -------------------------------------------------------------
int main() {
	//exercise1("text");
	//exercise2("text");
	exercise3();
}