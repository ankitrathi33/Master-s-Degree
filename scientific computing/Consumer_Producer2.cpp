// Includes
#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <sstream>
#include <chrono>
#include <atomic>
#include <fstream>

// Settings
#define TABLE_SIZE 10000
#define MAX_TABLES 100
#define NUM_PRODUCERS 1
#define NUM_CONSUMERS 1
constexpr std::atomic_int MAX_TABLES_PRODUCED = 10000;
constexpr std::atomic_int MAX_TABLES_CONSUMED = 10000;
std::atomic_int TABLES_PRODUCED = 0;
std::atomic_int TABLES_CONSUMED = 0;

//#define PRINT_YIELD_INFO
//#define PRINT_PROD_CONS_INFO
//#define WRITE_YIELD_INFO
//#define WRITE_PROD_CONS_INFO

#if defined WRITE_PROD_CONS_INFO || defined WRITE_YIELD_INFO
std::ofstream OUTPUT_FILE("output.txt");
#endif

// Types
typedef std::array<int, TABLE_SIZE> table;
typedef std::chrono::high_resolution_clock::time_point time_var;

// Auxiliaries
#define duration(a) std::chrono::duration_cast<std::chrono::microseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()


class Producer {
	public:
	Producer(Queue * Q) {
		this->Q = Q;
	}

	void Produce() {
		std::stringstream output;
		while (TABLES_PRODUCED < MAX_TABLES_PRODUCED) {
			if (this->Q->isFull()) {
				#ifdef WRITE_YIELD_INFO
				OUTPUT_FILE << "Producer thread " << std::this_thread::get_id() << ": yielded." << std::endl;
				#endif
				#ifdef PRINT_YIELD_INFO
				output << "Producer thread " << std::this_thread::get_id() << ": yielded." << std::endl;
				std::cout << output.str();
				#endif
				std::this_thread::yield();
			}
			else {
				this->Q->Enqueue(this->RandomTable());
				TABLES_PRODUCED++;
				#ifdef WRITE_PROD_CONS_INFO
				OUTPUT_FILE << "Producer thread " << std::this_thread::get_id() << ": produced a new table." << std::endl;
				#endif
				#ifdef PRINT_PROD_CONS_INFO
				//output << "Producer thread " << std::this_thread::get_id() << ": produced a new table." << std::endl;
				output.str("");
				output.clear();
				output << "Produced tables: " << TABLES_PRODUCED << std::endl;
				std::cout << output.str();
				#endif
			}
			
		}
	}

	private:
	Queue * Q;

	table RandomTable() {
		table result;
		for (int i = 0; i < TABLE_SIZE; i++) {
			result[i] = rand() % RAND_MAX;
		}
		return result;
	}
};

class Consumer {
	public:
	int sorted;

	Consumer(Queue * Q) {
		this->Q = Q;
		this->sorted = 0;
	}

	void Consume() {
		std::stringstream output;
		while (TABLES_CONSUMED < MAX_TABLES_CONSUMED) {
			if (this->Q->isEmpty()) {
				#ifdef WRITE_YIELD_INFO
				OUTPUT_FILE << "Consumer thread " << std::this_thread::get_id() << ": yielded." << std::endl;
				#endif
				#ifdef PRINT_YIELD_INFO
				output << "Consumer thread " << std::this_thread::get_id() << ": yielded." << std::endl;
				std::cout << output.str();
				#endif
				std::this_thread::yield();
			}
			else {
				table t;
				if (!this->Q->Dequeue(&t))
					continue;
				std::sort(std::begin(t), std::end(t));
				sorted++;
				TABLES_CONSUMED++;
				#ifdef WRITE_PROD_CONS_INFO
				OUTPUT_FILE << "Consumer thread " << std::this_thread::get_id() << ": consumed and sorted a table, for a total of: " << sorted << "." << std::endl;
				#endif
				#ifdef PRINT_PROD_CONS_INFO
				//output << "Consumer thread " << std::this_thread::get_id() << ": consumed and sorted a table, for a total of: " << sorted << "." << std::endl;
				output.str("");
				output.clear();
				output << "Consumed tables: " << TABLES_CONSUMED << std::endl;
				std::cout << output.str();
				#endif
			}
		}
	}

	private:
	Queue * Q;
};

int main() {

	//Queue tables(NUM_TABLES);
	//Producer producer(&tables);
	//Consumer consumer(&tables);
	//
	//std::thread producer_thread(std::bind(&Producer::Produce, producer));
	//std::thread consumer_thread(std::bind(&Consumer::Consume, consumer));
	//producer_thread.join();
	//consumer_thread.join();

	
	std::vector<std::thread> producer_threads;
	std::vector<std::thread> consumer_threads;
	std::vector<Producer> producers;
	std::vector<Consumer> consumers;

	Queue tables(MAX_TABLES);

	time_var start = timeNow();
	for (int i = 0; i < NUM_PRODUCERS; i++)
		producers.emplace_back(&tables);
	for (int i = 0; i < NUM_CONSUMERS; i++)
		consumers.emplace_back(&tables);

	for (auto& p : producers)
		producer_threads.emplace_back(std::bind(&Producer::Produce, p));
	for (auto& c : consumers)
		consumer_threads.emplace_back(std::bind(&Consumer::Consume, c));

	for (auto& t : producer_threads)
		t.join();
	for (auto& t : consumer_threads)
		t.join();

	#ifdef WRITE_PROD_CONS_INFO || WRITE_YIELD_INFO
	OUTPUT_FILE.close();
	#endif

	std::cout << "Test finished in: " << duration(timeNow() - start)/1000 << " ms" << std::endl;
	int i = 0;
	/*for (auto& c : consumers)
		std::cout << "Consumer " << i++ << " sorted " << c.sorted << " tables." << std::endl;*/
	std::cout << "Total tables produced: " << TABLES_PRODUCED << std::endl;
	std::cout << "Total tables consumed: " << TABLES_CONSUMED << std::endl;
}
