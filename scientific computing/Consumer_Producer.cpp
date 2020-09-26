#include <iostream>
#include <array>
#include <queue>
#include <mutex>
#include <thread>
#include <ctime>
#include <chrono>
#include <condition_variable>

template<typename T>
class queue_template
{
private:
	mutable std::mutex mut;
	std::queue<T> data_queue;
	std::condition_variable data_cond;

	std::condition_variable pop_cond;
	int queue_size;
	bool producent_alive;

public:
	//Initialise
	queue_template(int size)
	{
		queue_size = size;
		producent_alive = true;
	}
	
	queue_template(queue_template const& wsk)
	{
		std::lock_guard<std::mutex> lk(wsk.mut);
		data_queue = wsk.data_queue;
	}
	//When the queue is empty
	bool empty()
	{
		std::lock_guard<std::mutex> lk(mut);
		return data_queue.empty();
	}
	//To see wether the producer still exists (and produce arrays)
	void info(bool alive)
	{
		std::unique_lock<std::mutex> lk(mut);
		producent_alive = alive;
		pop_cond.notify_all();
	}

	void push(T  new_value)
	{
		std::unique_lock<std::mutex> lk(mut);
		//The Conditions if actual size is smaller or equal of the given size while initializing
		data_cond.wait(lk, [this] {return data_queue.size() < queue_size;  });
		data_queue.push(new_value);
		pop_cond.notify_one();
	}
	void wait_and_pop(T& value)
	{
		std::unique_lock<std::mutex> lk(mut);
		//The Conditions if the queue is not empty(or empty but still the producer is still there)
		pop_cond.wait(lk, [this] {return (!data_queue.empty() || (data_queue.empty() && !producent_alive)); });
		if (!producent_alive) return;
		value = data_queue.front();
		data_queue.pop();
		data_cond.notify_one();
	}
};
 //--------------------------------------------
class Consumer
{
private:
	int summary=0;
	queue_template<std::array<int, 100000>> *queue;
public:
	Consumer(queue_template<std::array<int, 100000>>& wsk)
	{
		queue = &wsk;
	}
	void run()
	{
		do
		{
			std::array<int, 100000> arr;
			queue->wait_and_pop(arr);
			std::sort(arr.begin(), arr.end()); //Sorts
			std::cout << adlerChecksum(arr) << std::endl; //Displays checksum by Adler algorythm
			summary++;
		} while (!queue->empty());
		std::cout <<"The number of the used arrays: "<< summary << std::endl;
	}

	unsigned long adlerChecksum(std::array<int, 100000> &arr)
	{
		int MOD = 65521;
		unsigned long a = 1;
		unsigned long b = 0;
		for (int i = 0; i < 100000; i++)
		{
			a = (a + arr[i]) % MOD;
			b = (b + a) % MOD;
		}
		return b << 16 | a;
	}
};
//--------------------------------------------

class Producer
{
private:
	queue_template<std::array<int, 100000>> *queue;
	int HowManyItemsNeed = 200;
public:
	Producer(queue_template<std::array<int, 100000>>& wsk)
	{
		queue = &wsk;
	}
	void run()
	{
		int i = 0;
		while (i < HowManyItemsNeed)
		{
			std::array<int, 100000> arr;
			for (int j = 0; j < arr.size(); j++)
				arr[j] = rand() % 10000 + 1;
			queue->push(arr);
			i++;
		}

		//When the queue is not empty
		while (!queue->empty())
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		//Notify the consumers when the production of array has been ended
		queue->info(false);
	}
};
//--------------------------------------------

int main()
{
	//auto threads = std::thread::hardware_concurrency();
	//Creating template
	queue_template<std::array<int, 100000>> *queue = new queue_template<std::array<int, 100000>>(200);

	Producer producer = Producer(*queue);
	std::thread producerThread(&Producer::run, &producer);

	//Creation of the vector of consumers
	std::vector<Consumer> consumers;
	std::vector<std::thread> consumerThreads;

	for (int i = 0; i < 8; i++)
		consumers.push_back(Consumer(*queue));

	for (auto& con : consumers)
		consumerThreads.emplace_back(&Consumer::run, &con);

	clock_t begin = clock();

	producerThread.join();
	for (auto& consumerThread : consumerThreads)
		consumerThread.join();
	
	clock_t end = clock();

	std::cout << "Time Elapsed: " << double(end - begin) / CLOCKS_PER_SEC << "ms" << std::endl;
	system("pause");
}