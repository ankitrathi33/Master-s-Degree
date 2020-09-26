// Francisco Bongiovanni


// Includes
#include "pch.h"
#include "mpi.h"
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
#include <chrono>


// Settings
constexpr auto TABLE_SIZE = 10000;
constexpr auto MAX_TABLES = 20;
constexpr auto MAX_TABLES_PRODUCED_CONSUMED = 1000;
constexpr auto BLOCKING_ROUTINES = true;
constexpr auto ROOT = 0; // Producer

int consumer_continue = true;
int total_consumed_tables;


// Types
typedef std::array<int, TABLE_SIZE + 1> table; // +1 so i can send a bool in the last index, used for stoping a consumer when ther are no more tables to be recieved 


// Functions
table RandomTable() {
	table result;
	for (int i = 0; i < TABLE_SIZE; i++) {
		result[i] = rand() % RAND_MAX;
	}
	return result;
}

void BProduce(int rank, int size) {
	int tables_produced = 0;
	int tables_consumed = 0;
	int current_tables = 0;
	table current;
	std::vector<table> tables;

	while (tables_produced < MAX_TABLES_PRODUCED_CONSUMED || tables_consumed < MAX_TABLES_PRODUCED_CONSUMED) {
		
		while (current_tables < MAX_TABLES && tables_produced < MAX_TABLES_PRODUCED_CONSUMED) {
			tables.insert(tables.begin(), RandomTable());
			current_tables++;
			tables_produced++;
		}

		for (int i = 1; i < size; i++) {
			if (tables.empty())
				break;

			//std::cout << "Process ROOT sending bool" << std::endl;
			current = tables.back();
			tables.pop_back();
			current[current.size() - 1] = (MAX_TABLES_PRODUCED_CONSUMED - tables_consumed - size) >= 0;
			//std::cout << "Process ROOT sending table" << std::endl;
			MPI_Send(&current[0], TABLE_SIZE + 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			
			tables_consumed++;
			current_tables--;
		}
	}

	for (int i = 1; i < size; i++) {
		consumer_continue = false;
		//std::cout << "ROOT send end bool to " << i << std::endl;
		MPI_Send(&consumer_continue, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
	}
	//std::cout << "Producer ended, tables produced: " << tables_produced << std::endl;
}

void NBProduce(int rank, int size) {
	int tables_produced = 0;
	int tables_sent = 0;
	int current_tables = 0;
	int send_completed = 1;
	table current;
	std::array<MPI_Request, 30> requests;
	MPI_Status status;
	std::vector<table> tables;

	requests.fill(MPI_REQUEST_NULL);

	while (tables_produced < MAX_TABLES_PRODUCED_CONSUMED || tables_sent < MAX_TABLES_PRODUCED_CONSUMED) {

		while (current_tables < MAX_TABLES && tables_produced < MAX_TABLES_PRODUCED_CONSUMED) {
			tables.push_back(RandomTable());
			current_tables++;
			tables_produced++;
		}

		for (int i = 1; i < size; i++) {

			if (tables.empty())
				break;	
			
			MPI_Test(&requests[i], &send_completed, &status);

			if (!send_completed)
				continue;

			current = tables.back();
			tables.pop_back();
			current[current.size() - 1] = 1;
			MPI_Isend(&current, TABLE_SIZE + 1, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i]);

			tables_sent++;
			current_tables--;
		}
	}

	for (int i = 1; i < size; i++) {
		current[current.size() - 1] = 0;
		MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
		MPI_Isend(&current, TABLE_SIZE + 1, MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i]);
	}

	std::cout << "Producer ended, tables produced: " << tables_produced << ", tables sent: " << tables_sent << std::endl;
}

int BConsume(int rank) {
	int tables_consumed = 0;
	table current;

	while (consumer_continue) {
		MPI_Recv(&current, TABLE_SIZE + 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		consumer_continue = current[current.size() - 1];
		std::sort(current.begin(), current.end() - 1);
		tables_consumed++;
	}

	std::cout << "Consumer " << rank << " ended, tables sorted: " << tables_consumed << std::endl;
	return tables_consumed;
}

int NBConsume(int rank) {
	int tables_consumed = 0;
	MPI_Request request = MPI_REQUEST_NULL;
	table current;
	table next;

	MPI_Irecv(&next, TABLE_SIZE + 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD, &request);	

	while (true) {
		MPI_Wait(&request, MPI_STATUS_IGNORE); // Wait for current
		std::copy(next.begin(), next.end(), current.begin()); // Get current
		consumer_continue = current[current.size() - 1];
		
		if (!consumer_continue)
			break;

		MPI_Irecv(&next, TABLE_SIZE + 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD, &request); // Ask for next
		
		std::sort(current.begin(), current.end() - 1);
		tables_consumed++;
	}

	std::cout << "Consumer " << rank << " ended, tables sorted: " << tables_consumed << std::endl;
	return tables_consumed;
}

void ProducerConsumer(int argc, char* argv[]) {
	int rank, size, tables_consumed = 0;
	double start;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(rank == ROOT)
		start = MPI_Wtime();

	if (rank == ROOT)
		BLOCKING_ROUTINES ? BProduce(rank, size) : NBProduce(rank, size);
	else
		tables_consumed = BLOCKING_ROUTINES ? BConsume(rank) : NBConsume(rank);

	MPI_Reduce(&tables_consumed, &total_consumed_tables, 1, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);

	if (rank == ROOT) {
		std::cout << "Total tables sorted: " << total_consumed_tables << std::endl;
		std::cout << "Test finished in: " << (MPI_Wtime() - start) * 1000 << " ms" << std::endl;
	}	

	MPI_Finalize();
}

int main(int argc, char* argv[]) {
	ProducerConsumer(argc, argv);
	return 0;
}