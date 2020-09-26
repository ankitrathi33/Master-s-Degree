#include "pch.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>


class Producer {
	public:
	Producer(Queue * Q) {
		this->Q = Q;
	}

	void Produce() {
		std::stringstream output;
		for (TABLES_PRODUCED; TABLES_PRODUCED < MAX_TABLES_PRODUCED; TABLES_PRODUCED++) {
			if (this->Q->isFull()) {
				TABLES_PRODUCED--;
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
				#ifdef WRITE_PROD_CONS_INFO
				OUTPUT_FILE << "Producer thread " << std::this_thread::get_id() << ": produced a new table." << std::endl;
				#endif
				#ifdef PRINT_PROD_CONS_INFO
				output << "Producer thread " << std::this_thread::get_id() << ": produced a new table." << std::endl;
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
	int sorted = 0;

	Consumer(Queue * Q) {
		this->Q = Q;
	}

	void Consume() {
		std::stringstream output;
		for (TABLES_CONSUMED; TABLES_CONSUMED < MAX_TABLES_CONSUMED; TABLES_CONSUMED++) {
			if (this->Q->isEmpty()) {
				TABLES_CONSUMED--;
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
				this->sorted++;
				#ifdef WRITE_PROD_CONS_INFO
				OUTPUT_FILE << "Consumer thread " << std::this_thread::get_id() << ": consumed and sorted a table, for a total of: " << sorted << "." << std::endl;
				#endif
				#ifdef PRINT_PROD_CONS_INFO
				output << "Consumer thread " << std::this_thread::get_id() << ": consumed and sorted a table, for a total of: " << sorted << "." << std::endl;
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

	std::cout << "Test finished in: " << duration(timeNow() - start) << " us" << std::endl;
	std::cout << "Total tables produced: " << TABLES_PRODUCED << std::endl;
	std::cout << "Total tables consumed: " << TABLES_CONSUMED << std::endl;
}